#!/bin/bash

usage () {
    echo "USAGE: $0 [--data-dir DATA_DIR] [--sync]";
    exit;
}

resolve () {
    if [ -z $1 ]
    then
        echo "invalid path to resolve";
        exit;
    fi
    local _pwd=$(pwd);
    local path=$1;
    cd ${path};
    local __pwd=$(pwd);
    cd ${_pwd};
    echo ${__pwd};
}

DATA_DIR=.
RESYNC=false
PYTHON=$(which python3)

while test $# != 0
do
    case $1 in
        --sync) RESYNC=true ;;
        --data-dir) 
            shift;
            DATA_DIR=$(resolve $1) ;;
        *) usage ;;
    esac
    shift;
done

if ${RESYNC}
then
    IMAGE_DIR=${DATA_DIR}/live/images;
    COLLECT_DIR=collected;
    IMAGE_STORE_URI=s3://driver-images-store;

    if [ ! -d ${IMAGE_DIR} ]
    then
        mkdir -p ${IMAGE_DIR};
    fi

    aws s3 sync ${IMAGE_STORE_URI} ${IMAGE_DIR} --exclude 864502035552157/* --exclude 12345*;

    for imei_dir in $(ls -1 ${IMAGE_DIR})
    do
        collect_dir=${IMAGE_DIR}/${imei_dir}/${COLLECT_DIR};
        if [ ! -d ${collect_dir} ]
        then
            mkdir ${collect_dir};
        fi

        for date_dir in $(find ${IMAGE_DIR}/${imei_dir} -mindepth 3 -type d)
        do
            cp ${date_dir}/*.jpeg ${collect_dir};
        done
    done
fi

export PYTHONPATH=${PYTHONPATH}:${DATA_DIR}/..;
${PYTHON} ${DATA_DIR}/scripts/imei.py ${DATA_DIR}/live/ --start 1548181810000

