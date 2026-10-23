function out = build_scara_ctc_simulink(dt_ctrl)
%BUILD_SCARA_CTC_SIMULINK  程序化搭建并运行 SCARA 的 CTC(+积分) 闭环 Simulink 模型
%
%   build_scara_ctc_simulink(1e-3)   % 控制周期 1 ms
%   build_scara_ctc_simulink(5e-3)   % 5 ms（验证离散化误差随 dt 线性增长）
%
% 结构：Clock → 参考轨迹(Ref) → Mux → 控制器(CTC) → 被控对象(Plant) → 两个积分器
%       记录到工作区：sim_q / sim_dq / sim_tau（Structure With Time）
if nargin < 1 || isempty(dt_ctrl), dt_ctrl = 1e-3; end
here = fileparts(mfilename('fullpath'));
addpath(here);

mdl = 'scara_ctc_sim';
if bdIsLoaded(mdl), close_system(mdl, 0); end
new_system(mdl);

add_block('simulink/Sources/Clock', [mdl '/Clock'], 'Position', [30 210 60 240]);
add_block('simulink/User-Defined Functions/MATLAB Fcn', [mdl '/Ref'], ...
          'MATLABFcn', 'scara_ctc_ref', 'OutputDimensions', '12', ...
          'Position', [100 190 220 250]);

% 控制器输入：u = [t; qd_ref(4); dqd_ref(4); ddqd_ref(4); q(4); dq(4)]（21×1）
add_block('simulink/Signal Routing/Mux', [mdl '/MuxCtrl'], 'Inputs', '4', ...
          'Position', [270 120 280 340]);
add_block('simulink/User-Defined Functions/MATLAB Fcn', [mdl '/CTC'], ...
          'MATLABFcn', 'scara_ctc_control', 'OutputDimensions', '4', ...
          'Position', [330 200 460 260]);

% 零阶保持：模拟真实数字控制器（控制周期 = dt_ctrl）。
% 不加这个块，ode4 会在每个子步调用控制器 → 等于"连续控制器"，误差会小几个数量级，
% 与 Python 侧的离散实现不可比。
add_block('simulink/Discrete/Zero-Order Hold', [mdl '/ZOH'], ...
          'SampleTime', num2str(dt_ctrl), 'Position', [480 205 510 255]);

add_block('simulink/Continuous/Integrator', [mdl '/Int_dq'], ...
          'InitialCondition', 'zeros(4,1)', 'Position', [520 200 550 250]);
add_block('simulink/Continuous/Integrator', [mdl '/Int_q'], ...
          'InitialCondition', '[0;0;0.05;0]', 'Position', [610 200 640 250]);

% 被控对象输入：u = [q(4); dq(4); tau(4)]（12×1）
add_block('simulink/Signal Routing/Mux', [mdl '/MuxPlant'], 'Inputs', '3', ...
          'Position', [700 150 710 330]);
add_block('simulink/User-Defined Functions/MATLAB Fcn', [mdl '/Plant'], ...
          'MATLABFcn', 'scara_plant_block', 'OutputDimensions', '4', ...
          'Position', [740 210 870 270]);

% 记录（向量直接进 To Workspace，避免一个端口接多根线）
add_block('simulink/Sinks/To Workspace', [mdl '/log_q'], 'VariableName', 'sim_q', ...
          'SaveFormat', 'Structure With Time', 'Position', [930 190 1000 220]);
add_block('simulink/Sinks/To Workspace', [mdl '/log_dq'], 'VariableName', 'sim_dq', ...
          'SaveFormat', 'Structure With Time', 'Position', [930 240 1000 270]);
add_block('simulink/Sinks/To Workspace', [mdl '/log_tau'], 'VariableName', 'sim_tau', ...
          'SaveFormat', 'Structure With Time', 'Position', [520 280 590 310]);

add_line(mdl, 'Clock/1', 'Ref/1', 'autorouting', 'on');
add_line(mdl, 'Clock/1', 'MuxCtrl/1', 'autorouting', 'on');
add_line(mdl, 'Ref/1', 'MuxCtrl/2', 'autorouting', 'on');
add_line(mdl, 'Int_q/1', 'MuxCtrl/3', 'autorouting', 'on');
add_line(mdl, 'Int_dq/1', 'MuxCtrl/4', 'autorouting', 'on');
add_line(mdl, 'MuxCtrl/1', 'CTC/1', 'autorouting', 'on');
add_line(mdl, 'CTC/1', 'ZOH/1', 'autorouting', 'on');
add_line(mdl, 'Int_dq/1', 'Int_q/1', 'autorouting', 'on');
add_line(mdl, 'Int_q/1', 'MuxPlant/1', 'autorouting', 'on');
add_line(mdl, 'Int_dq/1', 'MuxPlant/2', 'autorouting', 'on');
add_line(mdl, 'ZOH/1', 'MuxPlant/3', 'autorouting', 'on');
add_line(mdl, 'MuxPlant/1', 'Plant/1', 'autorouting', 'on');
add_line(mdl, 'Plant/1', 'Int_dq/1', 'autorouting', 'on');
add_line(mdl, 'Int_q/1', 'log_q/1', 'autorouting', 'on');
add_line(mdl, 'Int_dq/1', 'log_dq/1', 'autorouting', 'on');
add_line(mdl, 'CTC/1', 'log_tau/1', 'autorouting', 'on');

set_param(mdl, 'Solver', 'ode4', 'FixedStep', num2str(dt_ctrl), 'StopTime', '4.0');
save_system(mdl, fullfile(here, [mdl '.slx']));
out = sim(mdl);
% sim() 的结果默认只放在 SimulationOutput 里；这里回写到基础工作区，
% 方便脚本对比、也方便用户在命令行直接画图
try
    assignin('base', 'sim_q', out.sim_q);
    assignin('base', 'sim_dq', out.sim_dq);
    assignin('base', 'sim_tau', out.sim_tau);
catch ME
    warning('回写 sim_q/sim_dq/sim_tau 失败：%s', ME.message);
end
end
