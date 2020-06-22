//
// Created by dji on 2020/6/19.
//
#include <sys/types.h>
#include <sys/wait.h>
#include <cstdlib>
#include <cstdio>
#include <unistd.h>
#include <iostream>


void createsubprocess(int num){
    pid_t pid;
    int i{0};
    for(;i<num;++i){
        pid = fork();
        if(pid == 0 || -1 == pid){
            break;
        }
    }
    if(-1 == pid){
        exit(1);
    }
    else if(0 == pid){
        char *const ps_argv[] = {};
        int ret = execl("/home/dji/messi/docker_share_dir/cuda_example/build/src/MPS_test/mps_test","mps_test","","",NULL);
        if(ret < 0){
            std::cout << "execl subprocess err: " << ret << std::endl;
        }
        printf("子进程id=%d,其对应的父进程id=%d\n",getpid(),getppid());
        exit(0);
    }
    else{
        wait(NULL);
        printf("父进程id=%d\n",getpid());
        exit(0);
    }
}

int main(int argc ,char *argv[]){

    int num_process = 4;
    createsubprocess(num_process);
    return 0;
}