k = 10
i = len(out_selfconsistent["all_profiles"]) // 91 - k + 1
plt.semilogy(out_selfconsistent["cld"][-k]["temperature"], out_selfconsistent["cld"][-k]["pressure"], label="virga")
plt.semilogy(out_selfconsistent["all_profiles"][i*91:(i+1)*91], out_selfconsistent["pressure"], label="picaso")
plt.gca().invert_yaxis()
plt.legend()