#include "libspu/pir/trivial_symmetric_pir/trivial_spir.h"

#include "yacl/io/rw/csv_writer.h"

#include "libspu/pir/trivial_symmetric_pir/trivial_spir_components.h"
#include "libspu/psi/utils/csv_header_analyzer.h"
#include "libspu/psi/utils/serialize.h"
#include "libspu/psi/utils/utils.h"

namespace spu::pir {
using duration_millis = std::chrono::duration<double, std::milli>;

namespace {
std::vector<uint8_t> ReadEcSecretKeyFile(const std::string& file_path) {
  size_t file_byte_size = 0;
  try {
    file_byte_size = std::filesystem::file_size(file_path);
  } catch (std::filesystem::filesystem_error& e) {
    SPU_THROW("ReadEcSecretKeyFile {} Error: {}", file_path, e.what());
  }
  SPU_ENFORCE(file_byte_size == spu::psi::kEccKeySize,
              "error format: key file bytes is not {}", spu::psi::kEccKeySize);

  std::vector<uint8_t> secret_key(spu::psi::kEccKeySize);

  auto in =
      spu::psi::io::BuildInputStream(spu::psi::io::FileIoOptions(file_path));
  in->Read(secret_key.data(), spu::psi::kEccKeySize);
  in->Close();

  return secret_key;
}
}  // namespace

PirResultReport TrivialSpirFullyOnlineClient(
    const std::shared_ptr<yacl::link::Context>& lctx,
    const PirClientConfig& config) {
  // ====================== [step 0] sync meta info =================
  std::vector<std::string> id_columns;
  id_columns.insert(id_columns.end(), config.key_columns().begin(),
                    config.key_columns().end());

  // recv the byte length of the label (including the delimiter)
  size_t label_length = spu::psi::utils::DeserializeSize(
      lctx->Recv(lctx->NextRank(), fmt::format("label_byte_count")));

  // recv label columns
  yacl::Buffer label_columns_buffer =
      lctx->Recv(lctx->NextRank(), fmt::format("recv label columns name"));

  // initialize the output csv file
  std::vector<std::string> label_columns_name;
  spu::psi::utils::DeserializeStrItems(label_columns_buffer,
                                       &label_columns_name);
  yacl::io::Schema s;
  for (size_t i = 0; i < id_columns.size(); ++i) {
    s.feature_types.push_back(yacl::io::Schema::STRING);
  }
  for (size_t i = 0; i < label_columns_name.size(); ++i) {
    s.feature_types.push_back(yacl::io::Schema::STRING);
  }
  s.feature_names = id_columns;
  s.feature_names.insert(s.feature_names.end(), label_columns_name.begin(),
                         label_columns_name.end());

  yacl::io::WriterOptions w_op;
  w_op.file_schema = s;
  std::string query_output_file = config.output_path();
  auto out = spu::psi::io::BuildOutputStream(
      spu::psi::io::FileIoOptions(query_output_file));
  yacl::io::CsvWriter writer(w_op, std::move(out));
  writer.Init();

  auto step0_stats = lctx->GetStats();
  float step0_recv_mBytes = step0_stats->recv_bytes / (1000 * 1000.0);
  SPDLOG_INFO(
      "[step0 ]*** Client and server sync the label column names, and label "
      "byte count is {}; recv: {} mB",
      label_length, step0_recv_mBytes);

  // ==== [step 1] client receives server's OPRF values
  spu::psi::EcdhOprfPsiOptions psi_options;
  psi_options.link0 = lctx;
  psi_options.link1 = lctx->Spawn();
  psi_options.curve_type = spu::psi::CurveType::CURVE_FOURQ;
  std::shared_ptr<LabeledEcdhOprfPsiClient> labeled_dh_oprf_psi_client =
      std::make_shared<LabeledEcdhOprfPsiClient>(psi_options);

  labeled_dh_oprf_psi_client->SetLabelLength(label_length);

  const auto step2_start = std::chrono::system_clock::now();

  std::vector<std::string> server_ids;
  std::vector<std::string> server_labels;

  labeled_dh_oprf_psi_client->RecvFinalEvaluatedItems(&server_ids,
                                                      &server_labels);
  SPU_ENFORCE(server_ids.size() == server_labels.size());

  const auto step2_end = std::chrono::system_clock::now();
  const duration_millis step2_duration = step2_end - step2_start;
  auto step2_stats = lctx->GetStats();
  float step2_recv_mBytes = step2_stats->recv_bytes / (1000 * 1000.0);
  SPDLOG_INFO(
      "[step2, offline]*** Client receives server's evaluated items; duration: "
      "{} s, receives: {} mB",
      step2_duration.count() / 1000, step2_recv_mBytes);

  // ==== [step 2] client sends blinded items to the server
  std::unique_ptr<spu::psi::CsvBatchProvider> client_batch_provider =
      std::make_unique<spu::psi::CsvBatchProvider>(config.input_path(),
                                                   id_columns);

  const auto step3_start = std::chrono::system_clock::now();

  std::vector<std::string> client_ids;
  std::vector<std::string> client_blinded_ids;
  size_t self_items_count = labeled_dh_oprf_psi_client->SendBlindedItems(
      client_batch_provider, &client_ids);

  const auto step3_end = std::chrono::system_clock::now();
  const duration_millis step3_duration = step3_end - step3_start;

  SPDLOG_INFO(
      "[step3, online ]*** Client blinds its {} items and sends them to "
      "server; duration: {} s",
      client_ids.size(), step3_duration.count());

  // ==== [step 5] client receives blinded evaluated items, unblinds them
  const auto step5_start = std::chrono::system_clock::now();

  std::vector<std::string> client_evaluated_ids;
  std::vector<std::string> client_label_keys;
  std::vector<std::string> client_blinded_evaluated_ids(client_ids.size());
  std::vector<std::string> client_blinded_label_keys(client_ids.size());
  labeled_dh_oprf_psi_client->RecvEvaluatedItems(&client_evaluated_ids,
                                                 &client_label_keys);

  const auto step5_end = std::chrono::system_clock::now();
  const duration_millis step5_duration = step5_end - step5_start;
  auto step5_stats = lctx->GetStats();
  float step5_recv_mBytes =
      (step5_stats->recv_bytes) / (1000 * 1000.0) - step2_recv_mBytes;
  SPDLOG_INFO(
      "[step5, online ]*** Client receives {} blinded OPRF values from the "
      "server and unblinds them; duration: {} s, recv: {} mB",
      self_items_count, step5_duration.count() / 1000, step5_recv_mBytes);

  // === [step 6] client calculates the intersection and decrypt the label
  const auto step6_start = std::chrono::system_clock::now();
  std::vector<uint64_t> indices;
  std::vector<std::string> labels;
  std::shared_ptr<spu::psi::MemoryBatchProvider> server_batch_provider =
      std::make_shared<spu::psi::MemoryBatchProvider>(server_ids,
                                                      server_labels);
  std::tie(indices, labels) =
      labeled_dh_oprf_psi_client->FinalizeAndDecryptLabels(
          server_batch_provider, client_evaluated_ids, client_label_keys);

  const auto step6_end = std::chrono::system_clock::now();
  const duration_millis step6_duration = step6_end - step6_start;

  SPDLOG_INFO(
      "[step6, online] Client calculates the intersection by comparing its and "
      "the server's OPRF values; duration: {} s",
      step6_duration.count() / 1000);

  // ==== [step 7] client writes the query responses the file
  SPU_ENFORCE(indices.size() == labels.size());
  // sort labels by indices in ascending order
  std::vector<int> indexes(indices.size());
  std::iota(indexes.begin(), indexes.end(), 0);
  std::sort(indexes.begin(), indexes.end(),
            [&](int A, int B) -> bool { return indices[A] < indices[B]; });
  std::vector<std::string> sorted_labels(labels.size());
  for (size_t i = 0; i < indexes.size(); i++) {
    sorted_labels[i] = labels[indexes[i]];
  }

  std::vector<std::vector<std::string>> query_label_results(
      label_columns_name.size());
  for (size_t i = 0; i < sorted_labels.size(); ++i) {
    std::vector<std::string> result_labels =
        absl::StrSplit(sorted_labels[i], ",");
    SPU_ENFORCE(result_labels.size() == label_columns_name.size());
    for (size_t j = 0; j < result_labels.size(); ++j) {
      query_label_results[j].push_back(result_labels[j]);
    }
  }
  yacl::io::ColumnVectorBatch batch;
  batch.AppendCol(client_ids);
  for (size_t i = 0; i < label_columns_name.size(); ++i) {
    batch.AppendCol(query_label_results[i]);
  }
  writer.Add(batch);
  writer.Close();

  SPDLOG_INFO(
      "****** Client gets {} query responses. Write the responses to file {}",
      labels.size(), query_output_file);

  PirResultReport report;
  report.set_data_count(labels.size());

  yacl::link::AllGather(lctx, "task finished", "client and server sync");
  // wait the other party to exit safely
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  return report;
}

// TODO: the PirServer uses the PirSetupConfig as pass-in params
PirResultReport TrivialSpirFullyOnlineServer(
    const std::shared_ptr<yacl::link::Context>& lctx,
    const PirSetupConfig& config) {
  std::vector<std::string> id_columns;
  id_columns.insert(id_columns.end(), config.key_columns().begin(),
                    config.key_columns().end());

  // read column names from the csv file (note that column names can also be
  // obtained by config.label_columns())
  std::shared_ptr<spu::psi::CsvHeaderAnalyzer> csv_head_analyzer =
      std::make_shared<spu::psi::CsvHeaderAnalyzer>(config.input_path(),
                                                    id_columns);
  std::vector<std::string> label_columns;
  for (std::string field : csv_head_analyzer->headers()) {
    if (std::find(id_columns.begin(), id_columns.end(), field) ==
        id_columns.end()) {
      label_columns.push_back(field);
    }
  }

  // ==== [step 0]: sync meta info
  SPU_ENFORCE(label_columns.size() > 0);
  size_t label_length = config.label_max_len();

  // send the label byte count
  lctx->SendAsync(lctx->NextRank(),
                  spu::psi::utils::SerializeSize(label_length),
                  fmt::format("label_byte_count"));

  // send labels column name
  yacl::Buffer labels_buffer =
      spu::psi::utils::SerializeStrItems(label_columns);
  lctx->SendAsync(lctx->NextRank(), labels_buffer,
                  fmt::format("send label columns name"));

  SPDLOG_INFO("[ step0 ]*** Client and server sync the label column names");

  // ==== [step 1] server evaluates its items and sends them to the client
  spu::psi::EcdhOprfPsiOptions psi_options;
  psi_options.link0 = lctx;
  psi_options.link1 = lctx->Spawn();
  psi_options.curve_type = spu::psi::CurveType::CURVE_FOURQ;
  std::string id_key_path = fmt::format("{}-id", config.oprf_key_path());
  std::string label_key_path = fmt::format("{}-label", config.oprf_key_path());
  std::vector<uint8_t> id_server_private_key = ReadEcSecretKeyFile(id_key_path);
  std::vector<uint8_t> label_server_private_key =
      ReadEcSecretKeyFile(label_key_path);

  std::shared_ptr<LabeledEcdhOprfPsiServer> labeled_dh_oprf_psi_server =
      std::make_shared<LabeledEcdhOprfPsiServer>(
          psi_options, id_server_private_key, label_server_private_key,
          label_length);

  const auto step1_start = std::chrono::system_clock::now();
  std::shared_ptr<spu::psi::IBatchProvider> batch_provider =
      std::make_shared<spu::psi::CsvBatchProvider>(config.input_path(),
                                                   id_columns, label_columns);
  size_t self_items_count =
      labeled_dh_oprf_psi_server->FullEvaluateAndSend(batch_provider);

  const auto step1_end = std::chrono::system_clock::now();
  const duration_millis step1_duration = step1_end - step1_start;
  SPDLOG_INFO(
      "[step1, offline ]*** Server evaluates and sends {} items; duration: {} "
      "s",
      self_items_count, step1_duration.count() / 1000);

  // ==== [step 4] server receives the blinded items, evaluates and sends back
  const auto step4_start = std::chrono::system_clock::now();

  labeled_dh_oprf_psi_server->RecvBlindAndSendEvaluate();

  const auto step4_end = std::chrono::system_clock::now();
  const duration_millis step4_duration = step4_end - step4_start;
  auto step4_stats = lctx->GetStats();
  SPDLOG_INFO(
      "[step4, online ]*** Server receives client's blinded items, evaluates "
      "them and sends back to the client; duration: {} s, receives {} mB",
      step4_duration.count(), step4_stats->recv_bytes / (1000 * 1000.0));

  PirResultReport report;
  report.set_data_count(self_items_count);

  yacl::link::AllGather(lctx, "task finished", "client and server sync");
  // wait the other party to exit safely
  std::this_thread::sleep_for(std::chrono::milliseconds(1000));

  return report;
}
}  // namespace spu::pir