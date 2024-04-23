set(SCRIPT "${CMAKE_CURRENT_SOURCE_DIR}/../3rdparty/seasocks/scripts/gen_embedded.py")


set(EMBEDDED_FILES
        "${CMAKE_CURRENT_SOURCE_DIR}/app.js"
        "${CMAKE_CURRENT_SOURCE_DIR}/style.css"
        "${CMAKE_CURRENT_SOURCE_DIR}/_error.html"
        "${CMAKE_CURRENT_SOURCE_DIR}/error.css"
        "${CMAKE_CURRENT_SOURCE_DIR}/favicon.ico"
)


add_custom_command(OUTPUT Embedded.cpp
        COMMAND "${PYTHON_EXECUTABLE}" "${SCRIPT}" -o Embedded.cpp -f ${EMBEDDED_FILES}
        COMMENT "Generating embedded content"
        DEPENDS "${SCRIPT}" "${EMBEDDED_FILES}"
        )

add_library(embedded2 OBJECT Embedded.cpp)
target_include_directories(embedded2 PUBLIC $<TARGET_PROPERTY:seasocks,INTERFACE_INCLUDE_DIRECTORIES>)