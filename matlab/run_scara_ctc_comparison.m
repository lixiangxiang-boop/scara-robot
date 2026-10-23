function run_scara_ctc_comparison()
%RUN_SCARA_CTC_COMPARISON  Simulink CTC 闭环：与 Python(armLab) 对照 + 离散化扫描
%
% 运行：matlab -batch "run('matlab/run_scara_ctc_comparison.m')"
% 输出：outputs/data/scara_simulink_report.json、docs/generated/scara_simulink.md、图
here = fileparts(mfilename('fullpath'));
root = fileparts(here);
addpath(here);

ref = jsondecode(fileread(fullfile(root, 'outputs', 'data', 'scara_simulink_reference.json')));
t_ref = ref.t(:);  q_ref = ref.q;  tau_ref = ref.tau;

fprintf('\n===== Simulink CTC 闭环 vs Python(armLab) =====\n');
fprintf('场景：q0→q1 五次多项式 2 s；控制器 CTC；被控对象 SCARA 动力学 + 摩擦\n\n');

rows = struct('dt', {}, 'max_err', {}, 'final_err', {}, 'max_dev_python', {});
dts = [1e-3, 5e-3];
results = {};
for k = 1:numel(dts)
    dt = dts(k);
    build_scara_ctc_simulink(dt);
    qq = evalin('base', 'sim_q'); qdv = evalin('base', 'sim_dq');
    t = qq.time; Q = squeeze(qq.signals.values);
    if size(Q, 1) ~= numel(t), Q = Q.'; end
    % 参考：与 Python 同一轨迹公式
    q0 = [0; 0; 0.05; 0]; q1 = [deg2rad(60); deg2rad(-40); 0.15; deg2rad(90)];
    T = 2.0; tt = min(max(t / T, 0), 1); h = q1 - q0;
    s = 10*tt.^3 - 15*tt.^4 + 6*tt.^5;
    Qref = q0.' + h.' .* s;
    err = Q - Qref;
    % 与 Python 逐步对照（取公共时间网格）
    n = min(numel(t), numel(t_ref));
    dev = max(abs(Q(1:n, :) - q_ref(1:n, :)), [], 'all');
    results{end+1} = struct('dt', dt, 't', t, 'q', Q, 'err', err, 'dev', dev); %#ok<AGROW>
    if dt == 1e-3
        fprintf('dt = %-6.1e s : max|e| = %.3e rad, 终点误差 = %.3e, 与 Python(同为 1 ms) 最大偏差 = %.3e\n', ...
                dt, max(abs(err), [], 'all'), max(abs(err(end, :))), dev);
    else
        fprintf('dt = %-6.1e s : max|e| = %.3e rad, 终点误差 = %.3e（参考为 1 ms，不做逐步对照）\n', ...
                dt, max(abs(err), [], 'all'), max(abs(err(end, :))));
    end
end

% 离散化规律：误差应随 dt 近似线性下降
fprintf('\n离散化规律检查：dt 增大 5 倍 → max|e| 增大 %.1f 倍（理论≈5）\n', ...
        max(abs(results{2}.err), [], 'all') / max(abs(results{1}.err), [], 'all'));

out = struct();
out.scenario = ref.scenario;
out.simulink = struct('dt_1ms', results{1}, 'dt_5ms', results{2});
out.python_summary = ref.summary;
out.generated = datestr(now, 'yyyy-mm-dd HH:MM:SS');
outDir = fullfile(root, 'outputs', 'data');
if ~exist(outDir, 'dir'), mkdir(outDir); end
fid = fopen(fullfile(outDir, 'scara_simulink_report.json'), 'w', 'n', 'UTF-8');
fprintf(fid, '%s', jsonencode(struct('checks', {{'dt_1ms_max_err', max(abs(results{1}.err), [], 'all'), ...
    'dt_5ms_max_err', max(abs(results{2}.err), [], 'all'), ...
    'dev_vs_python_1ms', results{1}.dev, 'dev_vs_python_5ms', results{2}.dev, ...
    'python_max_err', ref.summary.max_err_rad}}), 'PrettyPrint', true));
fclose(fid);

% 图：轨迹与误差
f = figure('Visible', 'off', 'Position', [100 100 1100 620]);
subplot(2,2,1); plot(results{1}.t, results{1}.q, 'LineWidth', 1.1); grid on;
title('Simulink 闭环 q(t)（dt=1ms）'); xlabel('t / s'); ylabel('q');
subplot(2,2,2); plot(results{1}.t, results{1}.err, 'LineWidth', 1.1); grid on;
title('跟踪误差 e(t)'); xlabel('t / s'); ylabel('e / rad');
subplot(2,2,3); plot(t_ref, q_ref, '--', 'LineWidth', 1.0); grid on;
title('Python(armLab) 参考 q(t)'); xlabel('t / s'); ylabel('q');
subplot(2,2,4);
plot(results{1}.t, max(abs(results{1}.err), [], 2), 'LineWidth', 1.2); hold on;
plot(results{2}.t, max(abs(results{2}.err), [], 2), 'LineWidth', 1.2); grid on;
legend('dt = 1 ms', 'dt = 5 ms', 'Location', 'best');
title('离散化误差：dt 越大误差越大'); xlabel('t / s'); ylabel('max|e| / rad');
figDir = fullfile(root, 'outputs', 'figures');
if ~exist(figDir, 'dir'), mkdir(figDir); end
saveas(f, fullfile(figDir, 'scara_simulink_ctc.png')); close(f);

% markdown 报告
md = fullfile(root, 'docs', 'generated', 'scara_simulink.md');
fid = fopen(md, 'w', 'n', 'UTF-8');
fprintf(fid, '# P4S：SCARA Simulink CTC 闭环报告（自动生成）\n\n');
fprintf(fid, '> 由 `matlab/run_scara_ctc_comparison.m` 生成；MATLAB %s\n\n', version);
fprintf(fid, '场景：`q0 = [0; 0; 0.05; 0]` → `q1 = [60°; −40°; 0.15; 90°]`，2 s 五次多项式；\n');
fprintf(fid, '控制器：CTC（Kp=[60,60,400,20]，Kd=[12,12,40,6]）+ 摩擦前馈；被控对象：SCARA 动力学 + 粘性/库仑摩擦。\n\n');
fprintf(fid, '| 控制周期 | Simulink max\\|e\\| | 终点误差 | 与 Python 最大偏差 |\n|---|---|---|---|\n');
fprintf(fid, '| 1 ms | %.3e rad | %.3e | %.3e |\n', ...
        max(abs(results{1}.err), [], 'all'), max(abs(results{1}.err(end, :))), results{1}.dev);
fprintf(fid, '| 5 ms | %.3e rad | %.3e | %.3e |\n', ...
        max(abs(results{2}.err), [], 'all'), max(abs(results{2}.err(end, :))), results{2}.dev);
fprintf(fid, '\n- Python(armLab) 同一场景：max\\|e\\| = %.3e rad\n', ref.summary.max_err_rad);
fprintf(fid, '- 离散化规律：dt 放大 5 倍，误差放大约 %.1f 倍（与 e≈C·dt/Kp 一致）\n', ...
        max(abs(results{2}.err), [], 'all') / max(abs(results{1}.err), [], 'all'));
fprintf(fid, '\n图：`outputs/figures/scara_simulink_ctc.png`\n');
fclose(fid);
fprintf('\n报告：%s\n图：%s\n', md, fullfile(figDir, 'scara_simulink_ctc.png'));
end
