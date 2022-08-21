cd $PROJECT_ROOT

# create directory
mkdir examples/nn

# copy boilerplate CMakeLists.txt to `examples/nn`
cp examples/bfs/CMakeLists.txt examples/nn/CMakeLists.txt

# change APPLICATION_NAME from `bfs` to `nn`
sed -i "s/set(APPLICATION_NAME bfs)/set(APPLICATION_NAME nn)/" examples/nn/CMakeLists.txt