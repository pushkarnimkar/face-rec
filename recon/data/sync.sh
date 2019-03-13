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
    local path="$1";
    cd ${path};
    local __pwd=$(pwd);
    cd ${_pwd};
    echo ${__pwd};
}

DATA_DIR=.
BASEDIR=$(dirname "$0")
RESYNC=false
PYTHON=$(which python3)

while test $# != 0
do
    case "$1" in
        --sync) RESYNC=true ;;
        --data-dir) 
            shift;
            DATA_DIR=$(resolve "$1") ;;
        *) usage ;;
    esac
    shift;
done

IMAGE_DIR=${DATA_DIR}/images;
VAR_DIR=${DATA_DIR}/var;

populate () {
    IFS=$'\n'; set -f;
    IMAGE_DIR="${1}/images";
    CATEGORIES_DIR="${1}/categories/${2}/${3}";
    FILELIST="$(dirname ${0})/categories/${2}/${3}/filelist.txt";

    if [ ! -d ${CATEGORIES_DIR} ]
    then
        mkdir -p ${CATEGORIES_DIR};
    fi

    for ent in $(cat < "${FILELIST}")
    do
        imei=$(echo ${ent} | cut -d' ' -f1);
        epoch=$(echo ${ent} | cut -d' ' -f2);
        if [ -z ${imei} ]
        then
            continue;
        fi
        cp "${IMAGE_DIR}/${imei}/collected/${epoch}.jpeg" "${CATEGORIES_DIR}";
    done
}

if ${RESYNC}
then
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

if [ ! -d ${VAR_DIR} ]
then
    mkdir -p ${VAR_DIR};
fi

export PYTHONPATH=${PYTHONPATH}:${BASEDIR}/../../;
meta_cmd="${PYTHON} ${BASEDIR}/scripts/imei.py ${DATA_DIR} --start 1548181810000"
echo ${meta_cmd};
eval ${meta_cmd};

populate ${DATA_DIR} "whiteout" "positive";
populate ${DATA_DIR} "whiteout" "negative";

