# LINK-minimalRL-pytorch

Implementations of basic RL algorithms with minimal lines of codes! (PyTorch based)

* Each algorithm is complete within a single file.

* Length of each file is up to 100~150 lines of codes.

* Every algorithm can be trained within 30 seconds, even without GPU.

* Envs are fixed to "CartPole-v1". You can just focus on the implementations.

## Dependencies
1. PyTorch
2. OpenAI GYM
  - conda install -c conda-forge gym-all
3. lz4
  - pip install lz4
4. nvidia-ml-py3
  - pip install nvidia-ml-py3

## Git을 통한 초기화 방법 Permalink
1. 기존의 히스토리 삭제
$ rm -rf .git
2. 파일정리 후 새로운 git 설정
$ git init
$ git add .
$ git commit -m "first commit"
3. git 저장소 연결 후 강제 push
$ git remote add origin {git remote url}
$ git push -u --force origin master
   