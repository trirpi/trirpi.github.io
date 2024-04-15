+++
author = "Tristan Trouwen"
title = "Developing Audio Plugins with JUCE and Visual Studio Code"
date = "2024-04-14"
description = "This tutorial will help you to get up to speed developing a basic audio plugin using JUCE and CMake in VS Code in the simplest way possible."
tags = [
    "JUCE",
    "audio",
    "C++",
]
categories = [
    "tutorials",
]
#series = ["Themes Guide"]
#aliases = ["migrate-from-jekyl"]
+++

Developing audio plugins can be a daunting at first. Except for knowledge of C++, one has to have knowledge of digital signal processing, GUI development. However, the first obstacle is getting to know a good plugin development framework. JUCE is pretty much the industry standard in this regard. Unfortunately, the JUCE quickstart is not very beginner friendly. It will start to explain the Projucer, a custom system that creates XCode and Visual Studio build files. Although good IDEs, many people prefer VSCode or CLion. Although not clear from the website, these are also supported via CMake. [CMake](https://cmake.org/) markets itself as the de-facto standard for building C++ code and chances are you've come across it while browsing open source C++ projects. 

In 2020, the JUCE team [added support for CMake](https://forum.juce.com/t/native-built-in-cmake-support-in-juce/38700) which can be used instead of the Projucer.
To use CMake, you have to create a `CMakeLists.txt` file in your projects root directory. An example of such file specific to JUCE is given in the [JUCE Github's examples directory](https://github.com/juce-framework/JUCE/tree/master/examples/CMake/AudioPlugin).

## Setting up the directory

Create your project folder. I'll call mine `example-plugin`. Then clone the JUCE repository in there and copy the files in `JUCE/examples/CMake/AudioPlugin` to the `example-plugin` folder.
Your directory structure should look like this:
```
example-plugin/
├── JUCE/
├── CMakeLists.txt
├── PluginEditor.cpp
├── PluginEditor.h
├── PluginProcessor.cpp
└── PluginProcessor.h
```

## Setting up VS Code

Open the `example-plugin` folder in VS Code and install the C++ and Cmake plugins.

Now configure CMake by pressing `Ctrl+Shift+p` and typing `CMake: configure`. 
After pressing enter you will have to select a compiler and a build target. You can select `DEBUG` for now. 
If you press enter now, you will be greeted by the following error message:
```
CMake Error at CMakeLists.txt:40 (juce_add_plugin):
  Unknown CMake command "juce_add_plugin".
```

This is because we did not specify the location of the JUCE directory. Since it is simply in the project directory, we can uncomment the line that says `add_subdirectory(JUCE)` in `CMakeLists.txt`.
After doing this, the error should go away.

## Build the audio plugin

Now everything is set up, the last step is building the audio plugin. 
This can be done using the CMake build command.
To invoke this, press `Ctrl+Shift+p`, type `CMake: build target`, and then select `All`.
An executable and a vst3 plugin will be made in the build directory. You will have to go through some subdirectories to find it. 
The directory names depend on the compiler chosen, but my vst3 plugin was located in 
```
build\AudioPluginExample_artefacts\Debug\VST3\Audio Plugin Example.vst3\Contents\x86_64-win
```

## Next steps

Now you have a basic setup, you might want to flesh it out with some extra features. What about testing your code, using all your cores. In [the blog post by Melatonin.dev](https://melatonin.dev/blog/how-to-use-cmake-with-juce/), more advanced CMake features are explained.

Also check out the [PampleJUCE audio plugin template](https://github.com/sudara/pamplejuce). It contains several very useful features including the automatic creation of installers for Windows/Mac using Github actions.