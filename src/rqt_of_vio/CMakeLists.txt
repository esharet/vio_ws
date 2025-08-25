cmake_minimum_required(VERSION 3.8)
project(rqt_of_vio)

if(CMAKE_COMPILER_IS_GNUCXX OR CMAKE_CXX_COMPILER_ID MATCHES "Clang")
  add_compile_options(-Wall -Wextra -Wpedantic)
endif()

# find dependencies
find_package(ament_cmake REQUIRED)
find_package(ament_cmake_python REQUIRED)

ament_python_install_package(${PROJECT_NAME})

set(PLUGIN
  rqt_of_vio/of_vio_plugin.py
)


install(PROGRAMS
    ${PLUGIN}
DESTINATION lib/${PROJECT_NAME}
)

install(FILES
    plugin.xml
  DESTINATION share/${PROJECT_NAME}
)

install(DIRECTORY
  resource
DESTINATION share/${PROJECT_NAME}
)



ament_package()
