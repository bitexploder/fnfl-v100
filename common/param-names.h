#pragma once

// FNFL (FL2): the sampler and speculative-type names that argument parsing and the server read and print, moved out of
// sampling and speculative (cut with the old engine)

#include "common.h"

#include <string>
#include <vector>

char        common_sampling_type_to_chr(enum common_sampling_type cnstr);
std::string common_sampling_type_to_str(enum common_sampling_type cnstr);

std::vector<enum common_sampling_type> common_sampling_types_from_names(const std::vector<std::string> & names);
std::vector<enum common_sampling_type> common_sampling_types_from_chars(const std::string & chars);

// comma separated list the provided types
std::string common_spec_type_name_str(const std::vector<enum common_spec_type> & types);

// comma separated list of all types
const char * common_spec_all_types_str();

// parse user provided types
std::vector<enum common_spec_type> common_spec_types_from_names(const std::vector<std::string> & names);

// infer the spec types from the GGUF metadata of a draft model; empty if unknown
std::vector<enum common_spec_type> common_spec_types_from_gguf(const std::string & path);

// convert string to type
enum common_spec_type common_spec_type_from_name(const std::string & name);

// convert type to string
std::string common_spec_type_to_str(enum common_spec_type type);
