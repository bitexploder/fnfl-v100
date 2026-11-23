# FNFL: the server shell's resolved reasoning tags and its Qwen3-Coder parse of cut-off and closed reasoning, final
# and streamed, and the reasoning settings (server-reasoning-check); CPU only, no weights
set(TARGET fnfl-chat-check)

add_executable(${TARGET} fnfl-chat-check.cpp ${PROJECT_SOURCE_DIR}/tools/server/server-reasoning-check.cpp)
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/tools/server ${PROJECT_SOURCE_DIR})
target_link_libraries(${TARGET} PRIVATE server-context ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
