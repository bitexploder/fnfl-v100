#pragma once

// ME3 (the cleanup): the expert cache's host service as a protocol, free of CUDA so ex-cpu-test drives it with fakes.
// One thread (the service) runs a queue of window jobs and, with no job queued, the refill's copy batches. It owns the
// queue, the job in progress, the posted and served sequence per layer, the failure kept, and the stop and abandon
// protocol; the work it runs is two interfaces: the windows (the cache's window service) and the refill (the prompt
// loan's controller, which owns the refill's state).
//
// Producers and waits (each blocking wait names who advances it):
//   - the service waits for a job or a ready refill batch: post and post_pass (the decode thread), notify (a new
//     refill: the decode thread; a batch landed: the CUDA callback thread), abandon and stop.
//   - await, a job's descriptor: an eager window's event (the GPU, after the window's resolve and staging copies,
//     behind only earlier windows, whose flags the service wrote) or a replay window's stamp (the GPU replay, launched
//     by the decode thread after post_pass; its stamps come in layer order, each after the windows before it were
//     served). Ended by quit (abandon: the launch failed; stop).
//   - wait_idle and wait_layer (the decode thread): the service, which serves every queued job in order and ends a
//     refill batch after its current move while a job is queued or the decode thread waits (yield).
// Error unwind: the first failure is kept (a job's, a batch's, or the decode thread's post_failed); every job after
// it still waits for its descriptor (the window's resolve, which resets the flags, has run) and is then unblocked
// (its flags written), so no stream waits for ever; a job whose descriptor never comes (quit) is unblocked at once.
// The refill's failed is called once, outside the loop's lock (its waiters end).
//
// Lock order: the loop's mutex, then the refill's; never the reverse (the refill's callback releases its own lock
// before notify).

#include <atomic>
#include <condition_variable>
#include <cstddef>
#include <cstdint>
#include <deque>
#include <functional>
#include <mutex>
#include <string>
#include <thread>
#include <vector>

namespace fn::runtime::svc {

struct job {
    int32_t layer = 0, rows = 0;
    uint32_t seq = 0;
    int32_t par = 0;
    bool stamp = false; // a replay's window, waiting for its stamp (else for its event)
};

// the windows' work, every call on the service thread
class work {
  public:
    virtual ~work() = default;
    // job j's descriptor; false (err) when it never comes: quit set (the loop names the reason) or an error
    virtual bool await(const job & j, const std::atomic<bool> & quit, std::string & err) = 0;
    virtual bool serve(const job & j, std::string & err) = 0; // after await
    virtual void unblock(const job & j) = 0;                  // a failed job's flags written
};

// the refill's batches: ready and take under the loop's lock (take: [a, b) in flight), submit on the service thread
// outside it (yield true: end the batch after the current move); failed once, the first failure, outside the lock
class refill {
  public:
    virtual ~refill() = default;
    virtual bool ready() = 0;
    virtual void take(size_t & a, size_t & b) = 0;
    virtual bool submit(size_t a, size_t b, const std::function<bool()> & yield, std::string & err) = 0;
    virtual void failed(const std::string & err) = 0;
};

class loop {
  public:
    loop() = default;
    ~loop() { stop(); }
    loop(const loop &) = delete;
    loop & operator=(const loop &) = delete;

    void start(work * w, refill * r, int32_t layers);
    std::thread::native_handle_type handle() { return th_.native_handle(); }
    // an eager window (the decode thread, after its GPU work was enqueued)
    void post(int32_t layer, int32_t rows, int32_t par);
    // a replay's windows, one a layer in layer order at parity 0, each waiting for its stamp (before the launch)
    void post_pass(int32_t layers, int32_t rows);
    // a window whose GPU work was enqueued in part (its event recorded) and then failed: the failure kept, and the
    // job posted to be awaited and unblocked
    void post_failed(int32_t layer, int32_t rows, int32_t par, const std::string & err);
    void wait_idle();                // no job queued or in progress, no batch being submitted
    void wait_layer(int32_t layer);  // the layer's last posted job served
    void notify();                   // a refill's producer: a new refill, a landed batch
    void abandon(const std::string & why); // the replay's launch failed: waiting jobs end, and every later one
    // ME3 (load G ruling): the cache's GPU waits released: why kept as the failure (failed() true), waiting jobs end
    void release(const std::string & why);
    void stop();                     // stop: every queued job ended, the thread joined (idempotent)
    bool failed() const { return failed_.load(); }
    std::string error(); // the kept failure's message
    bool yield() const { return queued_.load() > 0 || idle_waiters_.load() > 0; }
    // TEST (ex-cpu-test): jobs served and unblocked, refill batches submitted
    int64_t served() const { return served_.load(); }
    int64_t unblocked() const { return unblocked_.load(); }
    int64_t batches() const { return batches_.load(); }

  private:
    void main();
    void record_failure(const std::string & e);

    work * w_ = nullptr;
    refill * r_ = nullptr;
    std::thread th_;
    std::mutex mu_;
    std::condition_variable cv_, idle_cv_;
    std::deque<job> queue_;
    bool busy_ = false, stop_ = false;
    std::atomic<bool> quit_{false};
    std::string abandon_; // (mu_)
    uint32_t seq_ = 0;
    std::vector<uint32_t> posted_, done_; // per layer (mu_)
    std::atomic<bool> failed_{false};
    std::string fail_err_; // (mu_)
    std::atomic<int32_t> queued_{0}, idle_waiters_{0};
    std::atomic<int64_t> served_{0}, unblocked_{0}, batches_{0};
};

} // namespace fn::runtime::svc
