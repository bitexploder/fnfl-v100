#pragma once

// VIS1 Stage B: the vision service on two cards without the text model (fnfl-drive vis-test DEV0 DEV1 MMPROJ IMAGE;
// DEV0 = DEV1 runs both logical cards on one device). The vision classes are reserved alone (loader reserve_vision,
// two read lanes) and the projector placed by vision_service::init. An image's prompt is read in two chunks that cut
// it in the middle. Checks, each bitwise:
//   - each card's supplied rows equal its half of the same image's rows from one card's whole encoder (Stage A's
//     path: all 27 blocks and the merger on DEV1), for both chunks, and in the other lane;
//   - the second chunk copies the held image without encoding it again; the other lane encodes it once;
//   - each card's rope planes and block planes equal image_rows::chunk's;
//   - refusals: an image the chunk reaches with no prepared image, and one whose digest differs from its span;
//   - slices (the job steps): 27 one-block encode_step calls give the same rows; a forget mid-encode, then the chunk
//     again, gives the same rows; two lanes queued encode in queue order, each into its own lane.
// Returns 0 for PASS.

#include <cstdint>

namespace fn::runtime {

int vision_service_test(int32_t dev0, int32_t dev1, const char * mmproj, const char * image);

}  // namespace fn::runtime
