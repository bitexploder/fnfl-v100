# dependencies

find_package(Threads REQUIRED)

# third-party

# ...

# flags

llama_add_compile_flags()

# tools

# FNFL (FL2): the server and its web UI only; the tools that drove the old engine are gone
if (LLAMA_BUILD_SERVER)
    add_subdirectory(ui)
    add_subdirectory(server)
    add_subdirectory(fnfl-chat-replay)
    add_subdirectory(fnfl-splice-test)
endif()
