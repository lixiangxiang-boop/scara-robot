%VERIFY_AGAINST_PYTHON P3：MATLAB 独立重算，与 Python(armLab) 参考数据逐项对比
%
% 运行（项目根目录）：matlab -batch "run('matlab/verify_against_python.m')"
here = fileparts(mfilename('fullpath'));
root = fileparts(here);
addpath(here);
refPath = fullfile(root, 'outputs', 'data', 'matlab_reference.json');
ref = jsondecode(fileread(refPath));

p = struct('L1', ref.params.L1, 'L2', ref.params.L2, ...
           'm1', ref.params.m1, 'm2', ref.params.m2, ...
           'lc1', ref.params.lc1, 'lc2', ref.params.lc2, ...
           'I1', ref.params.I1, 'I2', ref.params.I2, ...
           'gravity', ref.params.gravity(:));

N = ref.n_states;
err = struct('fk', 0, 'J', 0, 'M', 0, 'G', 0, 'Cdq', 0, 'tau', 0, 'ik', 0);
for i = 1:N
    q = ref.q(i,:)';  dq = ref.dq(i,:)';  ddq = ref.ddq(i,:)';
    err.fk  = max(err.fk,  max(abs(fk2r(q,p) - ref.xy(i,:)')));
    err.J   = max(err.J,   max(abs(jac2r(q,p) - squeeze(ref.J(i,:,:))), [], 'all'));
    err.M   = max(err.M,   max(abs(M2r(q,p) - squeeze(ref.M(i,:,:))), [], 'all'));
    err.G   = max(err.G,   max(abs(G2r(q,p) - ref.G(i,:)')));
    err.Cdq = max(err.Cdq, max(abs(Cdq2r(q,dq,p) - ref.Cdq(i,:)')));
    err.tau = max(err.tau, max(abs(tau2r(q,dq,ddq,p) - ref.tau(i,:)')));
    q_ik = ik2r(ref.xy(i,:)', p, 'down');
    err.ik = max(err.ik, max(abs(fk2r(q_ik,p) - ref.xy(i,:)')));
end

% 轨迹一致性检查（五次多项式）
[q_t, dq_t, ddq_t] = quintic2r(ref.traj_q(1,:)', ref.traj_q(end,:)', ...
                               ref.traj_t(end), ref.traj_t);
err.traj_q  = max(abs(q_t - ref.traj_q.'), [], 'all');
err.traj_dq = max(abs(dq_t - ref.traj_dq.'), [], 'all');
err.traj_ddq = max(abs(ddq_t - ref.traj_ddq.'), [], 'all');

% 输出报告
fprintf('\n===== P3 MATLAB vs Python(armLab) 验证 =====\n');
fprintf('状态组数: %d，轨迹点数: %d\n', N, numel(ref.traj_t));
names = fieldnames(err);
pass = true;
for k = 1:numel(names)
    v = err.(names{k});
    ok = v < 1e-10;
    pass = pass && ok;
    fprintf('  %-10s 最大偏差 %.3e  %s\n', names{k}, v, string(ok));
end
fprintf('结论: %s\n\n', string(pass));

out = struct('max_errors', err, 'pass', pass, 'n_states', N, ...
             'matlab_version', version, 'generated', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
outDir = fullfile(root, 'outputs', 'data');
if ~exist(outDir, 'dir'), mkdir(outDir); end
fid = fopen(fullfile(outDir, 'matlab_verification.json'), 'w', 'n', 'UTF-8');
fprintf(fid, '%s', jsonencode(out, 'PrettyPrint', true));
fclose(fid);

% 生成 markdown 报告
md = fullfile(root, 'docs', 'generated', 'matlab_verification.md');
fid = fopen(md, 'w', 'n', 'UTF-8');
fprintf(fid, '# P3：MATLAB 计算验证报告（自动生成）\n\n');
fprintf(fid, '> 由 `matlab/verify_against_python.m` 生成；MATLAB %s。\n\n', version);
fprintf(fid, '对照对象：`python/armlab`（解析闭式）导出的 %d 组随机状态 + 1 条轨迹。\n\n', N);
fprintf(fid, '| 检查项 | 最大偏差 | 判据 (<1e-10) |\n| --- | --- | --- |\n');
title_map = struct('fk','正运动学 FK','J','雅可比 J','M','质量矩阵 M','G','重力矩 G', ...
    'Cdq','科氏力矩 C·q̇','tau','逆动力学 τ','ik','逆运动学往返', ...
    'traj_q','轨迹 q(t)','traj_dq','轨迹 q̇(t)','traj_ddq','轨迹 q̈(t)');
for k = 1:numel(names)
    v = err.(names{k});
    fprintf(fid, '| %s | %.3e | %s |\n', title_map.(names{k}), v, ...
        string(v < 1e-10));
end
fprintf(fid, '\n**总结论：%s**\n', string(pass));
fclose(fid);
fprintf('报告: docs/generated/matlab_verification.md\n');
