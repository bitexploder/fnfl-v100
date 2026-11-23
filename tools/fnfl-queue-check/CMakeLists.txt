# FNFL: the server's task ids across the counter's wrap, through server_queue and server_response; host
# only, no model and no card
set(TARGET fnfl-queue-check)

add_executable(${TARGET} fnfl-queue-check.cpp)
target_include_directories(${TARGET} PRIVATE ${PROJECT_SOURCE_DIR}/tools/server ${PROJECT_SOURCE_DIR})
target_link_libraries(${TARGET} PRIVATE server-context ${CMAKE_THREAD_LIBS_INIT})
target_compile_features(${TARGET} PRIVATE cxx_std_17)
