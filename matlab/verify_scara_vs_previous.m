%VERIFY_SCARA_VS_PREVIOUS 用你已有的 SCARA MATLAB 模型对照 armLab（4 轴 R-R-P-R）
%
% 复用对象：…\ai学习框架\scara\matlab\scara_params.m / scara_dynamics.m
% 参考数据：outputs/data/scara_reference.json（由 armLab 生成）
% 运行：matlab -batch "run('matlab/verify_scara_vs_previous.m')"
here = fileparts(mfilename('fullpath'));
root = fileparts(here);
addpath(here);

prev_dir = 'C:\Users\42962\Documents\ai\ai学习框架\scara\matlab';
if ~exist(prev_dir, 'dir')
    error('未找到既有 SCARA MATLAB 模型：%s', prev_dir);
end
addpath(prev_dir);
p = scara_params();

ref = jsondecode(fileread(fullfile(root, 'outputs', 'data', 'scara_reference.json')));
N = ref.n_states;
err = struct('M', 0, 'Cdq', 0, 'G', 0, 'tau', 0, 'F', 0);
for i = 1:N
    st = ref.states(i);
    q = st.q(:);  qd = st.dq(:);  qdd = st.ddq(:);
    [M_p, Cqd_p, G_p, F_p] = scara_dynamics(q, qd, p);
    M_a = reshape(st.M, 4, 4);
    C_a = st.Cdq(:);
    G_a = st.G(:);
    tau_a = st.tau(:);
    err.M   = max(err.M,   max(abs(M_p - M_a), [], 'all'));
    err.Cdq = max(err.Cdq, max(abs(Cqd_p - C_a)));
    err.G   = max(err.G,   max(abs(G_p - G_a)));
    err.tau = max(err.tau, max(abs((M_p*qdd + Cqd_p + G_p) - tau_a)));
end
fprintf('\n===== SCARA：既有 MATLAB 模型 vs armLab 通用内核 =====\n');
fprintf('对照状态数：%d\n', N);
names = fieldnames(err);
pass = true;
for k = 1:numel(names)
    v = err.(names{k});
    ok = v < 1e-10;
    pass = pass && ok;
    fprintf('  %-5s 最大偏差 %.3e  %s\n', names{k}, v, string(ok));
end
fprintf('结论：%s（判据 <1e-10）\n\n', string(pass));

out = struct('max_errors', err, 'pass', pass, 'n_states', N, ...
             'previous_model', prev_dir, 'generated', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
outDir = fullfile(root, 'outputs', 'data');
if ~exist(outDir, 'dir'), mkdir(outDir); end
fid = fopen(fullfile(outDir, 'scara_matlab_verification.json'), 'w', 'n', 'UTF-8');
fprintf(fid, '%s', jsonencode(out, 'PrettyPrint', true));
fclose(fid);
n_ours = 0;
try
    ours = jsondecode(fileread(fullfile(outDir, 'scara_reference.json')));
    n_ours = ours.n_states;
catch
end
fprintf('已写出 outputs/data/scara_matlab_verification.json（armLab 状态数 %d）\n', n_ours);
