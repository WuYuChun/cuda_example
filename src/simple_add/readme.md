# 简介
进行简单的输入加法，为了测试一下kernel函数

# 用法
```
nvprof -m dram_write_throughput application_name
nvprof -m dram_read_throughput  application_name 
```

| 类型 | 耗时 ms | 读带宽 | 写带宽 |
| ----| ----    | ------|-------|
|cpu  |  9.87   |   ---    | ---      |
|gpu(一个线程）| 869.961   |  478.93MB/s  |  14.129MB/s   |
|gpu(256线程) |  6.217  | 475.97MB/s  |  1.9110GB/s   |
|gpu(多个block)|  1.861   |  1.4653GB/s  | 6.4845GB/s  |



![](/home/dji/messi/notes/2020-06-03-10-02-32-gpu_memory.png)

# 测试不同的性能指标

|                                                              | gpu(一个线程) | gpu(256线程) | gpu(多个block,多个线程) | 备注                                                         |
| ------------------------------------------------------------ | ------------- | ------------ | ----------------------- | ------------------------------------------------------------ |
| inst_per_warp                                                | 7.2352×1096   | 2.8266×148   | 115.000000              | 每一个warp中执行指令的平均数                                 |
| branch_efficiency                                            | 100%          | 100%         | 100%                    | 非分歧分支与总分支的比率，以百分比表示                       |
| warp_execution_efficiency                                    | 1/32=3.13%    | 100%         | 100%                    | 每一个warp中活动的线程 和一个warp中最大支持的线程的比值      |
| warp_nonpred_execution_efficiency                            | 3.08%         | 98.55%       | 99.13%                  | 这个是怎么理解呢？                                           |
| inst_replay_overhead                                         | 0.00          | 0.000004     | 0.000515                | 每一条指令的重复次数                                         |
| dram_read_throughput                                         | 478.93MB/s    | 475.97MB/s   | 1.4653GB/s              | 读取设备内存的吞吐量                                         |
| dram_write_throughput                                        | 14.129MB/s    | 1.9110GB/s   | 6.4845GB/s              | 写设备内存的吞吐量                                           |
| dram_read_transactions                                       | 13836201      | 16083        | 84191                   | 读取设备内存的事务                                           |
| dram_write_transactions                                      | 438971        | 380592       | 377884                  | 写设备内存的事务                                             |
| dram_utilization                                             | Low (1)       | Low (1)      | Low (1)                 | 设备内存的利用率（相比于峰值）                               |
| dram_read_bytes                                              | 427.95M       | 4.17M        | 1.015MB                 | 设备内存-->L2 总字节数                                       |
| dram_write_bytes                                             | 12.816        | 11.610MB     | 11.536MB                | L2---->设备内存 总字节数                                     |
| ecc_throughput                                               | 0.00b/s       | 0.00b/s      | 0.00b/s                 | ECC throughput from L2 to DRAM ？？？？                      |
| ecc_transactions                                             | 0             | 0            | 0                       | Number of ECC transactions between L2 and DRAM ？？？        |
| flop_hp_efficiency                                           | 0.00%         | 0.00%        | 0.00%                   | 与峰值半精度浮点运算的比率                                   |
| flop_sp_efficiency                                           | 0.00%         | 0.00%        | 0.12%                   |                                                              |
| flop_dp_efficiency                                           | 0.00%         | 0.00%        | 0.00%                   |                                                              |
| half_precision_fu_utilization                                | Idle (0)      | Idle (0)     | Idle (0)                | 多处理器功能单元中执行16位浮点指令的使用级别，它们以0到10的比例 |
| single_precision_fu_utilization                              | Low (1)       | Low (1)      | Low (1)                 |                                                              |
| double_precision_fu_utilization                              | Idle (0)      | Idle (0)     | Idle (0)                |                                                              |
| achieved_occupancy                                           | 1.56%         | 12.49%       | 92.66%                  | 每个活动周期的平均活动warp数与多处理器上支持的最大warp数之比 |
| ipc                                                          | 0.047271      | 0.235430     | 0.060666                | 每个周期执行的指令                                           |
| issued_ipc                                                   | 0.046864      | 0.240890     | 0.060726                | 每个周期发出的指令                                           |
| sm_efficiency                                                | 4.99%         | 4.98%        | 99.79%                  | 特定多处理器上至少一次激活的时间百分比                       |
| shared_load_transactions_per_request                         | 0.00          | 0.00         | 0.00                    | 每个共享内存读取执行的共享内存负载事务的平均数量             |
| shared_store_transactions_per_request                        | 0.00          | 0.00         | 0.00                    | 每个共享内存存储执行的共享内存存储事务的平均数量             |
| shared_store_transactions                                    | 0             | 0            | 0                       | 写入共享内存事务的数量                                       |
| shared_load_transactions                                     | 0             | 0            | 0                       | 读取共享内存事务的数量                                       |
| shared_load_throughput                                       | 0.0           | 0.0          | 0.0                     | 读取共享内存的吞吐量                                         |
| shared_store_throughput                                      | 0.0           | 0.0          | 0.0                     | 写入共享内存的吞吐量                                         |
| shared_efficiency                                            | 0%            | 0%           | 0%                      | 请求的共享内存吞吐量与所需的共享内存吞吐量之比，以百分比表示 |
| shared_utilization                                           | Idle (0)      | Idle (0)     | Idle (0)                | 共享内存的利用率                                             |
| local_load_transactions_per_request                          |               |              |                         |                                                              |
| local_store_transactions_per_request                         |               |              |                         |                                                              |
| local_load_transactions                                      |               |              |                         |                                                              |
| local_store_transactions                                     |               |              |                         |                                                              |
| local_hit_rate                                               |               |              |                         |                                                              |
| local_memory_overhead                                        |               |              |                         | L1和L2缓存之间的本地内存流量与总内存流量之比，以百分比表示   |
| local_load_throughput                                        | 0.00000B/s    | 0.00000B/s   | 0.00000B/s              |                                                              |
| local_store_throughput                                       | 0.00000B/s    | 0.00000B/s   | 0.00000B/s              |                                                              |
| ldst_issued                                                  | 3145731       | 393248       | 557056                  | 已发射的本地，全局，共享和纹理内存加载和存储指令的数量       |
| ldst_executed                                                | 3145731       | 98336        | 262144                  | 已执行的本地，全局，共享和纹理内存加载和存储指令的数量       |
| inst_executed_local_loads                                    | 0             |              |                         |                                                              |
| inst_executed_local_stores                                   |               | 0            |                         |                                                              |
| gld_transactions_per_request                                 | 4.000514      | 16.000031    | 16.000031               | 每次全局内存加载事务的平均执行次数。                         |
| gst_transactions_per_request                                 | 1             | 4            | 4                       | 每个全局内存存储区平均执行的全局内存存储区交易次数           |
| gld_transactions                                             |               |              |                         |                                                              |
| gst_transactions                                             |               |              |                         |                                                              |
| sysmem_read_transactions                                     |               |              |                         |                                                              |
| sysmem_write_transactions                                    |               |              |                         |                                                              |
| l2_read_transactions                                         |               |              |                         |                                                              |
| l2_write_transactions                                        |               |              |                         |                                                              |
| global_hit_rate                                              |               |              |                         |                                                              |
| gld_requested_throughput                                     |               |              |                         |                                                              |
| gst_requested_throughput                                     |               |              |                         |                                                              |
| gld_throughput                                               |               |              |                         |                                                              |
| gst_throughput                                               |               |              |                         |                                                              |
| tex_cache_hit_rate                                           |               |              |                         |                                                              |
| l2_tex_read_hit_rate                                         |               |              |                         |                                                              |
| l2_tex_write_hit_rate                                        |               |              |                         |                                                              |
| tex_cache_throughput                                         |               |              |                         |                                                              |
| l2_tex_read_throughput                                       |               |              |                         |                                                              |
| l2_tex_write_throughput                                      |               |              |                         |                                                              |
| l2_read_throughput                                           |               |              |                         |                                                              |
| l2_write_throughput                                          |               |              |                         |                                                              |
| sysmem_read_throughput                                       |               |              |                         |                                                              |
| sysmem_write_throughput                                      |               |              |                         |                                                              |
| gld_efficiency                                               |               |              |                         |                                                              |
| gst_efficiency                                               |               |              |                         |                                                              |
| tex_cache_transactions                                       |               |              |                         |                                                              |
| flop_count_dp                                                |               |              |                         |                                                              |
| flop_count_dp_add                                            |               |              |                         |                                                              |
| flop_count_dp_fma                                            |               |              |                         |                                                              |
| flop_count_dp_mul                                            |               |              |                         |                                                              |
| flop_count_sp                                                |               |              |                         |                                                              |
| flop_count_sp_add                                            |               |              |                         |                                                              |
| flop_count_sp_fma                                            |               |              |                         |                                                              |
| flop_count_sp_mul                                            |               |              |                         |                                                              |
| flop_count_sp_special                                        |               |              |                         |                                                              |
| inst_executed                                                |               |              |                         |                                                              |
| inst_issued                                                  |               |              |                         |                                                              |
| sysmem_utilization                                           |               |              |                         |                                                              |
| stall_inst_fetch                                             |               |              |                         |                                                              |
| stall_exec_dependency                                        |               |              |                         |                                                              |
| stall_memory_dependency                                      |               |              |                         |                                                              |
| stall_texture                                                |               |              |                         |                                                              |
| stall_sync                                                   |               |              |                         |                                                              |
| stall_other                                                  |               |              |                         |                                                              |
| stall_constant_memory_dependency                             |               |              |                         |                                                              |
| stall_pipe_busy                                              |               |              |                         |                                                              |
| inst_fp_32                                                   |               |              |                         |                                                              |
| inst_fp_64                                                   |               |              |                         |                                                              |
| inst_integer                                                 |               |              |                         |                                                              |
| inst_bit_convert                                             |               |              |                         |                                                              |
| inst_control                                                 |               |              |                         |                                                              |
| inst_compute_ld_st                       inst_misc inst_inter_thread_communication                     issue_slots                       cf_issued                     cf_executed             atomic_transactions atomic_transactions_per_request<br/><br/>            l2_atomic_throughput<br/><br/>          l2_atomic_transactions<br/><br/>        l2_tex_read_transactions<br/><br/>           stall_memory_throttle<br/><br/>              stall_not_selected<br/><br/>       l2_tex_write_transactions<br/><br/>                   flop_count_hp<br/><br/>               flop_count_hp_add<br/><br/>               flop_count_hp_mul<br/><br/>               flop_count_hp_fma<br/><br/>                      inst_fp_16<br/><br/>         sysmem_read_utilization<br/><br/>        sysmem_write_utilization<br/><br/>     pcie_total_data_transmitted<br/><br/>        pcie_total_data_received<br/><br/>      inst_executed_global_loads<br/><br/>      <br/><br/>      inst_executed_shared_loads<br/><br/>     inst_executed_surface_loads<br/><br/>     inst_executed_global_stores<br/><br/><br/>     inst_executed_shared_stores<br/><br/>    inst_executed_surface_stores<br/><br/>    inst_executed_global_atomics<br/><br/> inst_executed_global_reductions<br/><br/>   inst_executed_surface_atomics<br/><br/>inst_executed_surface_reductions<br/><br/>    inst_executed_shared_atomics<br/><br/>           inst_executed_tex_ops<br/><br/>            l2_global_load_bytes<br/><br/>             l2_local_load_bytes<br/><br/>           l2_surface_load_bytes<br/><br/>     l2_local_global_store_bytes<br/><br/>       l2_global_reduction_bytes<br/><br/>    l2_global_atomic_store_bytes<br/><br/>          l2_surface_store_bytes<br/><br/>      l2_surface_reduction_bytes<br/><br/>   l2_surface_atomic_store_bytes<br/><br/>            global_load_requests<br/><br/>             local_load_requests<br/><br/>           surface_load_requests<br/><br/>           global_store_requests<br/><br/>            local_store_requests<br/><br/>          surface_store_requests<br/><br/>          global_atomic_requests<br/><br/>       global_reduction_requests<br/><br/>         surface_atomic_requests<br/><br/>      surface_reduction_requests<br/><br/>               sysmem_read_bytes<br/><br/>              sysmem_write_bytes<br/><br/>                 l2_tex_hit_rate<br/><br/>           texture_load_requests<br/><br/>           unique_warps_launched issue_slot_utilization eligible_warps_per_cycle                 tex_utilizationon:The utilization level of thcf_fu_utilization          special_fu_utilization              tex_fu_utilization |               |              |                         |                                                              |
|                                                              |               |              |                         |                                                              |
|                                                              |               |              |                         |                                                              |
|                                                              |               |              |                         |                                                              |
|                                                              |               |              |                         |                                                              |
|                                                              |               |              |                         |                                                              |
|                                                              |               |              |                         |                                                              |
|                                                              |               |              |                         |                                                              |