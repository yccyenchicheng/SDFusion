#!/bin/sh
#
# Copyright (c) 2005-2018 Intel Corporation
#
# Licensed under the Apache License, Version 2.0 (the "License");
# you may not use this file except in compliance with the License.
# You may obtain a copy of the License at
#
#     http://www.apache.org/licenses/LICENSE-2.0
#
# Unless required by applicable law or agreed to in writing, software
# distributed under the License is distributed on an "AS IS" BASIS,
# WITHOUT WARRANTIES OR CONDITIONS OF ANY KIND, either express or implied.
# See the License for the specific language governing permissions and
# limitations under the License.
#
#
#
#

get_library_directory () {
    gcc_version_full=$(gcc --version | grep "gcc" | egrep -o " [0-9]+\.[0-9]+\.[0-9]+.*" | sed -e "s/^\ //")
    if [ $? -eq 0 ]; then
        gcc_version=$(echo "$gcc_version_full" | egrep -o "^[0-9]+\.[0-9]+\.[0-9]+")
    fi
    case "${gcc_version}" in
    4.[7-9]*|[5-9]* )
        lib_dir="gcc4.7";;
    4.[4-6]* )
        lib_dir="gcc4.4";;
    * )
        lib_dir="gcc4.1";;
    esac
    echo $lib_dir
}

# Parsing script arguments
# Arg1 represents target architecture. Its possible values are 'ia32' or 'intel64',
# default value equals to the value of $COMPILERVARS_ARCHITECTURE environment variable.

TBB_TARGET_ARCH=

if [ -n "${COMPILERVARS_ARCHITECTURE}" ]; then
    TBB_TARGET_ARCH=$COMPILERVARS_ARCHITECTURE
fi

if [ -n "$1" ]; then
    TBB_TARGET_ARCH=$1
fi

if [ -n "${TBB_TARGET_ARCH}" ]; then
    if [ "$TBB_TARGET_ARCH" != "ia32" -a "$TBB_TARGET_ARCH" != "intel64" ]; then
        echo "ERROR: Unknown switch '$TBB_TARGET_ARCH'. Accepted values: ia32, intel64"
        TBB_TARGET_ARCH=
        return 1;
    fi
else
    echo "ERROR: Architecture is not defined. Accepted values: ia32, intel64"
    return 1;
fi

# Arg2 represents target platform. Its possible values are 'android' or 'linux'.
# If $COMPILERVARS_PLATFORM environment variable is defined,
# the default value of $TBB_TARGET_PLATFORM equals to its value.
# Otherwise it equals to 'linux'.
if [ "$2" = "linux" -o "$2" = "android" ]; then
    TBB_TARGET_PLATFORM=$2
elif [ "$COMPILERVARS_PLATFORM" = "linux" -o "$COMPILERVARS_PLATFORM" = "android" ]; then
    TBB_TARGET_PLATFORM=$COMPILERVARS_PLATFORM
else
    TBB_TARGET_PLATFORM="linux"
fi

# Arg3 represents TBBROOT detection method. Its possible value is 'auto_tbbroot'. In which case
# the environment variable TBBROOT is detected automatically by using the script directory path.
TBBROOT=SUBSTITUTE_INSTALL_DIR_HERE
if [ -n "${BASH_SOURCE}" ]; then
    if [ "$3" = "auto_tbbroot" ]; then
       TBBROOT=$(cd $(dirname ${BASH_SOURCE}) && pwd -P)/..
    fi
fi

LIBTBB_NAME="libtbb.so.2"
if [ "$TBB_TARGET_PLATFORM" != "android" ]; then
    which gcc >/dev/null 2>&1
    if [ $? -eq 0 ]; then
        library_directory="$TBB_TARGET_ARCH/$(get_library_directory)"
    else
        echo "WARNING: 'gcc' was not found"
        library_directory="$TBB_TARGET_ARCH/gcc4.1"
    fi
elif [ "$TBB_TARGET_PLATFORM" = "android" ]; then
    if [ "$TBB_TARGET_ARCH" = "ia32" ]; then
        library_directory="$TBB_TARGET_PLATFORM"
    elif [ "$TBB_TARGET_ARCH" = "intel64" ]; then
        library_directory="$TBB_TARGET_PLATFORM/x86_64"
    fi
    LIBTBB_NAME="libtbb.so"
else
    library_directory=""
fi

if [ -e "$TBBROOT/lib/$library_directory/$LIBTBB_NAME" ]; then
    export TBBROOT
    if [ -z "${LD_LIBRARY_PATH}" ]; then
        LD_LIBRARY_PATH="$TBBROOT/lib/$library_directory"; export LD_LIBRARY_PATH
    else
        LD_LIBRARY_PATH="$TBBROOT/lib/$library_directory:${LD_LIBRARY_PATH}"; export LD_LIBRARY_PATH
    fi
    if [ -z "${LIBRARY_PATH}" ]; then
        LIBRARY_PATH="$TBBROOT/lib/$library_directory"; export LIBRARY_PATH
    else
        LIBRARY_PATH="$TBBROOT/lib/$library_directory:${LIBRARY_PATH}"; export LIBRARY_PATH
    fi
    if [ -z "${CPATH}" ]; then
        CPATH="${TBBROOT}/include"; export CPATH
    else
        CPATH="${TBBROOT}/include:$CPATH"; export CPATH
    fi
else
    echo "ERROR: $LIBTBB_NAME library does not exist in $TBBROOT/lib/$library_directory."
    return 2
fi
