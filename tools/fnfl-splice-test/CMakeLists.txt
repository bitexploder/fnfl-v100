# FNFL (BG2): the walk that splices a slot's generated answers back into a chat prompt; CPU only, no model
set(TARGET fnfl-splice-test)

add_executable(${TARGET} fnfl-splice-test.cpp ${PROJECT_SOURCE_DIR}/tools/server/server-splice.cpp)
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/tools/server)
target_compile_features(${TARGET} PRIVATE cxx_std_17)
target_link_libraries(${TARGET} PRIVATE vendor::nlohmann)  # APF1: tu.writer_pause_ms's JSON parse (header only)
