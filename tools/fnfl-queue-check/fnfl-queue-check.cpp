// fnfl-queue-check (FBH1 B1): the server's task ids across the counter's wrap. The queue's loop runs on its own
// thread as in the server; tasks are posted and answered through server_queue and server_response with the counter
// started just below its wrap and one long-lived id held over it. An id is never negative, never the reserved
// no-result id and never one still queued or waiting for its result; the loop's NEXT_RESPONSE turn draws none.
//
// usage: fnfl-queue-check. Exit status 0 when every check passes.

#include "server-queue.h"
#include "server-task.h"

#include <atomic>
#include <cstdio>
#include <mutex>
#include <set>
#include <string>
#include <thread>
#include <vector>

namespace {

int g_fail = 0;

void report(bool ok, const std::string & what) {
    std::printf("queue-check: %s %s\n", ok ? "PASS" : "FAIL", what.c_str());
    g_fail += ok ? 0 : 1;
}

} // namespace

int main() {
    server_queue    q;
    server_response r;
    q.set_results(&r);

    std::mutex       m;
    std::vector<int> held;    // ids whose answer waits for a release
    std::atomic<int> turns{ 0 };
    std::atomic<int> cancels{ 0 };
    const auto       answer = [&](int id) {
        auto res = std::make_unique<server_task_result_metrics>();
        res->id  = id;
        r.send(std::move(res));
    };
    q.on_new_task([&](server_task && task, bool) {
        switch (task.type) {
            case SERVER_TASK_TYPE_COMPLETION: // stands for a long request: answered at the release
                {
                    std::unique_lock<std::mutex> lock(m);
                    held.push_back(task.id);
                }
                break;
            case SERVER_TASK_TYPE_SLOT_GET: // the release
                {
                    std::vector<int> ids;
                    {
                        std::unique_lock<std::mutex> lock(m);
                        ids.swap(held);
                    }
                    for (const int id : ids) {
                        answer(id);
                    }
                    answer(task.id);
                }
                break;
            case SERVER_TASK_TYPE_NEXT_RESPONSE:
                turns += 1;
                break;
            case SERVER_TASK_TYPE_CANCEL:
                cancels += 1;
                break;
            default:
                answer(task.id);
                break;
        }
        return true;
    });
    q.on_update_slots([]() {});
    std::thread loop([&]() { q.start_loop(); });

    const int      start = server_queue::id_max - 40;
    std::set<int>  live; // ids drawn and not yet done
    std::set<int>  seen;
    bool           valid = true, unique = true, answered = true;
    int            lowest = INT_MAX, highest = -1, n = 0;
    const auto     draw   = [&]() {
        const int id = q.get_new_id();
        valid        = valid && id >= 0 && id <= server_queue::id_max;
        unique       = unique && live.count(id) == 0;
        lowest       = std::min(lowest, id);
        highest      = std::max(highest, id);
        live.insert(id);
        seen.insert(id);
        ++n;
        return id;
    };
    const auto round_trip = [&](server_task_type type) {
        const int id = draw();
        r.add_waiting_task_id(id);
        server_task task(type);
        task.id = id;
        q.post(std::move(task));
        const server_task_result_ptr res = r.recv(id);
        answered                         = answered && res != nullptr && res->id == id;
        r.remove_waiting_task_id(id);
        live.erase(id);
    };

    q.set_next_id(start);
    const uint64_t drawn0 = q.ids_drawn();

    // the long-lived task: drawn just below the wrap, posted, waiting for its result until the end
    const int long_id = draw();
    r.add_waiting_task_id(long_id);
    {
        server_task task(SERVER_TASK_TYPE_COMPLETION);
        task.id = long_id;
        q.post(std::move(task));
    }

    // 200 tasks posted and completed across the wrap (ids id_max - 39 .. id_max, then 0 ..)
    for (int i = 0; i < 200; ++i) {
        round_trip(SERVER_TASK_TYPE_METRICS);
    }
    report(valid && unique && answered && highest == server_queue::id_max && lowest == 0 && seen.count(-1) == 0,
           "200 tasks posted and completed across the wrap from " + std::to_string(start) + ": every id in 0 .. " +
               std::to_string(server_queue::id_max) + ", highest " + std::to_string(highest) + " then lowest " +
               std::to_string(lowest) + ", none negative, each answered under its own id");

    // the batch post fills unset ids from the same counter: three cancels over the wrap, no assertion
    {
        q.set_next_id(server_queue::id_max);
        std::vector<server_task> cs;
        for (int i = 0; i < 3; ++i) {
            server_task c(SERVER_TASK_TYPE_CANCEL);
            c.id_target = 12345 + i; // nothing queued under these
            cs.push_back(std::move(c));
        }
        q.post(std::move(cs), true);
        const int next = draw();
        live.erase(next);
        round_trip(SERVER_TASK_TYPE_METRICS); // the queue has run the cancels by the time this is answered
        report(next == 2 && cancels.load() == 3,
               "a batch of 3 tasks with unset ids at the wrap takes " + std::to_string(server_queue::id_max) +
                   ", 0 and 1; the next id is " + std::to_string(next) + ", " + std::to_string(cancels.load()) +
                   " run");
    }

    // the counter comes round to the long-lived id while it still waits: skipped
    {
        q.set_next_id(long_id);
        const int a = draw();
        live.erase(a);
        report(a == long_id + 1 && unique,
               "the counter back at the waiting id " + std::to_string(long_id) + " gives " + std::to_string(a) +
                   " (the id in use is skipped)");
    }

    // an id still queued (a deferred task nobody waits for) is skipped too
    {
        const int d = draw();
        server_task task(SERVER_TASK_TYPE_METRICS);
        task.id = d;
        q.defer(std::move(task));
        q.set_next_id(d);
        const int a = draw();
        live.erase(a);
        report(a == d + 1 && unique,
               "the counter back at the deferred task's id " + std::to_string(d) + " gives " + std::to_string(a));
        q.pop_deferred_task(-1); // it runs; nobody waits for its answer
        live.erase(d);
    }

    // the slots' turn wake-up takes the reserved id and draws nothing
    {
        const uint64_t before = q.ids_drawn();
        for (int i = 0; i < 1000; ++i) {
            server_task task(SERVER_TASK_TYPE_NEXT_RESPONSE);
            task.id = server_queue::id_no_result;
            q.post(std::move(task));
        }
        round_trip(SERVER_TASK_TYPE_METRICS);
        const uint64_t after = q.ids_drawn();
        report(turns.load() == 1000 && after - before == 1 && seen.count(server_queue::id_no_result) == 0,
               "1000 NEXT_RESPONSE posts under the reserved id " + std::to_string(server_queue::id_no_result) +
                   ": " + std::to_string(turns.load()) + " run, " + std::to_string(after - before) +
                   " id drawn (the round trip after them)");
    }

    // the long-lived task completes under its own id, after everything that crossed the wrap
    {
        const int rel = draw();
        r.add_waiting_task_id(rel);
        server_task task(SERVER_TASK_TYPE_SLOT_GET);
        task.id = rel;
        q.post(std::move(task));
        const server_task_result_ptr a = r.recv(long_id);
        const server_task_result_ptr b = r.recv(rel);
        r.remove_waiting_task_id(long_id);
        r.remove_waiting_task_id(rel);
        live.erase(long_id);
        live.erase(rel);
        const bool free_again = !r.is_waiting(long_id);
        q.set_next_id(long_id);
        const int again = q.get_new_id();
        report(a != nullptr && a->id == long_id && b != nullptr && b->id == rel && free_again && again == long_id,
               "the long-lived task " + std::to_string(long_id) + " is answered under its id at the end; released, "
               "the id is drawn again (" + std::to_string(again) + ")");
    }

    const uint64_t drawn = q.ids_drawn() - drawn0;
    report(valid && unique && answered && live.empty(),
           std::to_string(n) + " ids drawn by the test (" + std::to_string(drawn) + " by the queue's count): no "
           "negative id, no collision with a live id, no assertion");

    q.terminate();
    loop.join();
    std::printf("queue-check: %s (%d failed)\n", g_fail == 0 ? "PASS" : "FAIL", g_fail);
    return g_fail == 0 ? 0 : 1;
}
