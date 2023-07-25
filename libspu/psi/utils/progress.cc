// Copyright 2023 Ant Group Co., Ltd.
//
// Licensed under the Apache License, Version 2.0 (the "License");
// you may not use this file except in compliance with the License.
// You may obtain a copy of the License at
//
//   http://www.apache.org/licenses/LICENSE-2.0
//
// Unless required by applicable law or agreed to in writing, software
// distributed under the License is distributed on an "AS IS" BASIS,
// WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
// See the License for the specific language governing permissions and
// limitations under the License.

#include "libspu/psi/utils/progress.h"

#include <mutex>
#include <numeric>

#include "fmt/format.h"

namespace spu::psi {

Progress::Progress(std::string description)
    : description_(description),
      percentage_(0),
      mode_(Mode::kSingle),
      done_(false) {}

void Progress::Update(size_t percentage) {
  percentage_.store(std::min<size_t>(percentage, 100));
}

void Progress::Done() {
  if (IsDone()) {
    return;
  }

  if (mode_.load() != Mode::kSingle) {
    std::unique_lock lock(rw_mutex_);

    for (size_t i = 0; i < sub_progresses_.size(); i++) {
      sub_progresses_[i]->Done();
    }
  }

  percentage_.store(100);
  done_.store(true);
}

bool Progress::IsDone() const { return done_.load(); }

Progress::Data Progress::Get() {
  if (mode_.load() == Mode::kSingle) {
    Data out;
    auto p = percentage_.load();
    out.total = 1;
    out.running = p > 0 && p < 100 ? 1 : 0;
    out.finished = p >= 100 ? 1 : 0;
    out.percentage = p;
    if (description_.empty()) {
      out.description = fmt::format("{}%", p);
    } else {
      out.description = fmt::format("{}, {}%", description_, p);
    }
    return out;
  }

  std::shared_lock r_lock(rw_mutex_);

  size_t total_weight =
      std::accumulate(weights_.begin(), weights_.end(), size_t(0));
  size_t total_percent = 0;
  Data out;

  std::string sub_desc;
  out.total = weights_.size();
  for (size_t i = 0; i < sub_progresses_.size(); i++) {
    auto sub_data = sub_progresses_[i]->Get();
    total_percent += weights_[i] * sub_data.percentage;

    out.total--;
    out.total += sub_data.total;
    out.finished += sub_data.finished;
    out.running += sub_data.running;

    // get running job's or last job's description
    if (sub_desc.empty() &&
        (!sub_progresses_[i]->IsDone() || i == sub_progresses_.size() - 1)) {
      sub_desc = sub_data.description;
    }
  }
  out.percentage = total_percent / (total_weight == 0 ? 1 : total_weight);

  if (mode_.load() == Mode::kParallel) {
    sub_desc = fmt::format("{} jobs running ({} / {})", out.running,
                           out.finished, out.total);
  }

  if (description_.empty()) {
    out.description = sub_desc;
  } else if (sub_desc.empty()) {
    out.description = description_;
  } else {
    out.description = description_ + ", " + sub_desc;
  }

  return out;
}

void Progress::SetWeights(std::vector<size_t> weights, Mode mode) {
  std::unique_lock lock(rw_mutex_);
  mode_.store(mode);
  weights_ = std::move(weights);
}

void Progress::SetSubJobCount(size_t count, Mode mode) {
  std::unique_lock lock(rw_mutex_);
  mode_.store(mode);
  weights_.resize(count, 1);
}

std::shared_ptr<Progress> Progress::AddSubProgress(
    const std::string& description) {
  std::unique_lock lock(rw_mutex_);

  auto p = std::make_shared<Progress>(description);
  sub_progresses_.push_back(p);
  return p;
}

std::shared_ptr<Progress> Progress::NextSubProgress(
    const std::string& description) {
  std::unique_lock lock(rw_mutex_);

  if (!sub_progresses_.empty()) {
    auto current = sub_progresses_.back();
    current->Done();
  }
  auto p = std::make_shared<Progress>(description);
  sub_progresses_.push_back(p);
  return p;
}

}  // namespace spu::psi