launch:
th export.lua {model} {1,3,512,512}

python import.py

test:
`bash test/test_all.sh` to run all tests, `bash test/test_net.sh net_name` to run one test on `net_name`, in that case `test/make_net_name.lua` should exist.
