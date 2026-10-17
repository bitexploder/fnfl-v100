# FNFL (TC1): replay raw generated text through the server's chat parsing and streaming code; CPU only, no model
set(TARGET fnfl-chat-replay)

add_executable(${TARGET} fnfl-chat-replay.cpp)
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/tools/server ${PROJECT_SOURCE_DIR})
target_link_libraries(${TARGET} PRIVATE server-context ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
