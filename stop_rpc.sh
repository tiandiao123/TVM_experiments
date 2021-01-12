
ps aux | grep rpc_server | awk '{print $2}' | xargs kill
