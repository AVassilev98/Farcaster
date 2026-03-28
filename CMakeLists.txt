cmake_minimum_required(VERSION 3.24)
project(Farcaster LANGUAGES C)

set(CMAKE_C_STANDARD 17)
set(CMAKE_C_STANDARD_REQUIRED ON)
set(CMAKE_EXPORT_COMPILE_COMMANDS ON)

if(APPLE AND NOT DEFINED ENV{VULKAN_SDK})
    execute_process(
        COMMAND brew --prefix vulkan-loader
        OUTPUT_VARIABLE VULKAN_LOADER_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    execute_process(
        COMMAND brew --prefix molten-vk
        OUTPUT_VARIABLE MOLTENVK_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    execute_process(
        COMMAND brew --prefix vulkan-headers
        OUTPUT_VARIABLE VULKAN_HEADERS_PREFIX
        OUTPUT_STRIP_TRAILING_WHITESPACE
        ERROR_QUIET
    )
    if(VULKAN_LOADER_PREFIX)
        set(Vulkan_LIBRARY     "${VULKAN_LOADER_PREFIX}/lib/libvulkan.dylib"  CACHE PATH "" FORCE)
        set(Vulkan_INCLUDE_DIR "${VULKAN_HEADERS_PREFIX}/include"             CACHE PATH "" FORCE)
        list(APPEND CMAKE_BUILD_RPATH "${VULKAN_LOADER_PREFIX}/lib")
    elseif(MOLTENVK_PREFIX)
        set(Vulkan_LIBRARY     "${MOLTENVK_PREFIX}/lib/libMoltenVK.dylib" CACHE PATH "" FORCE)
        set(Vulkan_INCLUDE_DIR "${MOLTENVK_PREFIX}/libexec/include"       CACHE PATH "" FORCE)
        list(APPEND CMAKE_BUILD_RPATH "${MOLTENVK_PREFIX}/lib")
    endif()
endif()
find_package(Vulkan REQUIRED)

add_subdirectory(external/Photon)

file(GLOB_RECURSE SOURCES src/*.c)
add_executable(${PROJECT_NAME} ${SOURCES})

target_include_directories(${PROJECT_NAME} PRIVATE
    ${CMAKE_CURRENT_SOURCE_DIR}/include
)

target_link_libraries(${PROJECT_NAME} PRIVATE
    photon
)


set(SHADER_SOURCES
    shaders/triangle.slang
)

set(SHADER_OUTPUT_DIR ${CMAKE_CURRENT_BINARY_DIR}/shaders)
file(MAKE_DIRECTORY ${SHADER_OUTPUT_DIR})

foreach(SHADER ${SHADER_SOURCES})
    get_filename_component(SHADER_NAME ${SHADER} NAME)
    string(REPLACE ".slang" ".spv" OUTPUT_NAME ${SHADER_NAME})

    set(OUTPUT_FILE ${SHADER_OUTPUT_DIR}/${OUTPUT_NAME})

    add_custom_command(
        OUTPUT  ${OUTPUT_FILE}
        COMMAND ${SLANGC} ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER} -o ${OUTPUT_FILE} -target spirv
        DEPENDS ${CMAKE_CURRENT_SOURCE_DIR}/${SHADER}
        COMMENT "Compiling ${SHADER}"
    )

    list(APPEND SHADER_OUTPUTS ${OUTPUT_FILE})
endforeach()

add_custom_target(${PROJECT_NAME}Shaders ALL DEPENDS ${SHADER_OUTPUTS})
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}Shaders)

target_compile_definitions(${PROJECT_NAME} PRIVATE
    SHADER_DIR="${SHADER_OUTPUT_DIR}"
    ASSET_DIR="${CMAKE_CURRENT_SOURCE_DIR}/assets"
)

# ---- Asset downloads --------------------------------------------------------
set(ASSET_DIR ${CMAKE_CURRENT_SOURCE_DIR}/assets)
set(SPONZA_DIR ${ASSET_DIR}/sponza)
set(SPONZA_STAMP ${SPONZA_DIR}/.downloaded)
set(CURTAINS_STAMP ${SPONZA_DIR}/.curtains_downloaded)

add_custom_command(
    OUTPUT  ${SPONZA_STAMP}
    COMMAND ${CMAKE_COMMAND} -E make_directory ${SPONZA_DIR}
    COMMAND curl -L -o ${SPONZA_DIR}/sponza_base.zip
            "https://cdrdv2.intel.com/v1/dl/getContent/830833"
    COMMAND ${CMAKE_COMMAND} -E chdir ${SPONZA_DIR}
            ${CMAKE_COMMAND} -E tar xf ${SPONZA_DIR}/sponza_base.zip --format=zip
    COMMAND ${CMAKE_COMMAND} -E rm ${SPONZA_DIR}/sponza_base.zip
    COMMAND ${CMAKE_COMMAND} -E touch ${SPONZA_STAMP}
    COMMENT "Downloading Intel Sponza base scene"
)

add_custom_command(
    OUTPUT  ${CURTAINS_STAMP}
    DEPENDS ${SPONZA_STAMP}
    COMMAND curl -L -o ${SPONZA_DIR}/curtains.zip
            "https://cdrdv2.intel.com/v1/dl/getContent/726650"
    COMMAND ${CMAKE_COMMAND} -E chdir ${SPONZA_DIR}
            ${CMAKE_COMMAND} -E tar xf ${SPONZA_DIR}/curtains.zip --format=zip
    COMMAND ${CMAKE_COMMAND} -E rm ${SPONZA_DIR}/curtains.zip
    COMMAND ${CMAKE_COMMAND} -E touch ${CURTAINS_STAMP}
    COMMENT "Downloading Intel Sponza colorful curtains add-on"
)

add_custom_target(${PROJECT_NAME}Assets DEPENDS ${SPONZA_STAMP} ${CURTAINS_STAMP})
add_dependencies(${PROJECT_NAME} ${PROJECT_NAME}Assets)
