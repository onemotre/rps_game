# rps_game

## Build

- C++ main

    ```bash
    mkdir ./build && cd ./build
    cmake ..
    make -j
    ```

- python 

    ```bash
    pip install -r requirements.txt
    ```

    

## Dependence

- main.cpp
  - g++ >= 11.4
  - cmake >= 3.1
  - openvino-runtime
  - opencv >= 4.8

- python script
  - openvino-python
  - opencv>=4.8
  - pytorch
  - onnxruntime==1.18.0



## Data Source

robotflow: [link](https://universe.roboflow.com/roboflow-58fyf/rock-paper-scissors-sxsw/dataset/14#)
