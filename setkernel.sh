echo 0 > /proc/sys/kernel/perf_event_paranoid
echo 0 > /proc/sys/kernel/kptr_restrict 
sysctl -w kernel.numa_balancing=0
