#include <iostream>

/**
 * Record multiplication operatioms in the offline phase.
 * Provide interface for both protocol implementation side and application side.
 * 
 * Static function:
 *  - void OfflineRecorder::StartRecorder();
 *   Start recording the multiplications happened in the offline phase.
 *   i.e. Creates a OfflineRecorder instance and binds it to OfflineRecorder::instance.
 *  - void OfflineRecorder::StopRecorder();
 *   Stop recording the multiplications happened in the offline phase.
 *   i.e. Deletes the OfflineRecorder instance, unbinds it from OfflineRecorder::instance and prints the recorded data.
 *  - void OfflineRecorder::RecordMult(size_t size);
 *   Record a multiplication operation with the given size.
 */
class OfflineRecorder {
public:
  static void StartRecorder() {
    if (instance == nullptr) {
      std::cout << "OfflineRecorder instance created" << std::endl;
      instance = new OfflineRecorder();
    } else {
      std::cout << "OfflineRecorder instance already exists" << std::endl;
      throw "OfflineRecorder instance already exists";
    }
  }

  static void StopRecorder() {
    if (instance != nullptr) {
      std::cout << "OfflineRecorder instance deleted" << std::endl;
      delete instance;
      instance = nullptr;
    } else {
      std::cout << "OfflineRecorder instance does not exist" << std::endl;
      throw "OfflineRecorder instance does not exist";
    }
  }

  static void RecordMult(size_t num_element, size_t size) {
    if (instance != nullptr) {
      instance->mult_count += num_element;
      instance->mult_comm += size;
    }
  }

  static void RecordAsyncComm(size_t num_element, size_t size) {
    if (instance != nullptr) {
      instance->async_comm_count += num_element;
      instance->async_comm_size += size;
    }
  }

  static void PrintRecord() {
    if (instance != nullptr) {
      std::cout << "+ Offline communication records + " << std::endl;
      std::cout << "  Multiplications: " << instance->mult_count << std::endl;
      std::cout << "    Communication: " << instance->mult_comm << std::endl;
      std::cout << "  Async Communication: " << instance->async_comm_count << std::endl;  
      std::cout << "    Communication: " << instance->async_comm_size << std::endl;
      std::cout << "  Totals: " << std::endl;
      std::cout << "    Comm: " << instance->mult_comm + instance->async_comm_size << std::endl;

    }
  }

private:
  static OfflineRecorder* instance;

  OfflineRecorder() = default;
  ~OfflineRecorder() {
    OfflineRecorder::PrintRecord();
  };

  // delete copy consrutctor and assignment operator
  OfflineRecorder(const OfflineRecorder&) = delete;
  OfflineRecorder& operator=(const OfflineRecorder&) = delete;
  // delete move constructor and assignment operator
  OfflineRecorder(OfflineRecorder&&) = delete;
  OfflineRecorder& operator=(OfflineRecorder&&) = delete;

  size_t mult_count = 0;
  size_t mult_comm = 0;
  size_t async_comm_count = 0;
  size_t async_comm_size = 0;
};
