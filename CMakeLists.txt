cmake_minimum_required(VERSION 4.0)
project(DL)

set(CMAKE_CXX_STANDARD 20)

add_executable(DL
        main.cpp
        NN.cpp
        saveData.cpp)

target_include_directories(DL PRIVATE ${CMAKE_SOURCE_DIR}/include)

# copie automatique du dossier web
add_custom_target(copy_web ALL
        COMMAND ${CMAKE_COMMAND} -E copy_directory
        ${CMAKE_SOURCE_DIR}/web
        ${CMAKE_CURRENT_BINARY_DIR}/web
        COMMENT "Copying web folder to build directory"
)

add_dependencies(DL copy_web)

#Compile pour windows
#x86_64-w64-mingw32-g++ ../DL/*.cpp -I../DL/include -o DL_Windows.exe -std=c++17 -lws2_32 -static-libgcc -static-libstdc++ && mkdir -p web && cp -R ../DL/web/* ./web/
#x86_64-w64-mingw32-g++ ../DL/*.cpp \-I../DL/include -o DL_Windows.exe \-std=c++17 -lws2_32 -static -static-libgcc -static-libstdc++