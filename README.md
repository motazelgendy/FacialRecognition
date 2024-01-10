# FacialRecognition
## This Tools using Emgucv as OpencCV Wrapper 

## this tool can be used as Face Matching Tool by installing it as a Docker Container

# Installing on Linux (Tested on Ubuntu 22.0 Jammy Jellyfish)
## git Emgucv 
```
git clone https://github.com/emgucv/emgucv emgucv
```
## Go to emgucv directory
```
cd emgucv
```
## Got to the configuration folder
```
cd platforms/ubuntu/22.04
```
## Installing the prerequisites. This only needs to be run once. You can install them by running
```
./apt_install_dependency
```
## in case u faced any runtime error go to libcvextern.so dir and run 
```
ldd libcvextern.so
```
