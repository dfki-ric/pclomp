# Unit tests, laid out like PCL's (test/registration): one gtest executable per file, the data
# file passed as a command-line argument. Enable with -DBUILD_TESTING=ON, run with ctest.
find_package(GTest REQUIRED)
find_package(PCL REQUIRED COMPONENTS io filters)

# pclomp_add_test(<name> FILES <sources> ARGUMENTS <args>), modeled on PCL's PCL_ADD_TEST
function(pclomp_add_test name)
	cmake_parse_arguments(ARG "" "" "FILES;ARGUMENTS" ${ARGN})
	add_executable(${name} ${ARG_FILES})
	target_link_libraries(${name} PRIVATE
		registration
		${PCL_IO_LIBRARIES}
		${PCL_FILTERS_LIBRARIES}
		GTest::gtest
		OpenMP::OpenMP_CXX
	)
	add_test(NAME ${name} COMMAND ${name} ${ARG_ARGUMENTS})
endfunction()

set(PCLOMP_TEST_CLOUD ${PROJECT_SOURCE_DIR}/data/251370668.pcd)

pclomp_add_test(test_gicp_omp FILES test_gicp_omp.cpp ARGUMENTS ${PCLOMP_TEST_CLOUD})
pclomp_add_test(test_ndt_omp FILES test_ndt_omp.cpp ARGUMENTS ${PCLOMP_TEST_CLOUD})
