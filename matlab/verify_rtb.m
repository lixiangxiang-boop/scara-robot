%VERIFY_RTB 用 Robotics System Toolbox 的 rigidBodyTree 独立建模，做第三方交叉验证
%
% 与 armLab 对比：FK（末端位置）、质量矩阵 M、逆动力学 τ（含重力/科氏）。
% 运行：matlab -batch "run('matlab/verify_rtb.m')"
here = fileparts(mfilename('fullpath'));
root = fileparts(here);
ref = jsondecode(fileread(fullfile(root, 'outputs', 'data', 'matlab_reference.json')));
p = struct('L1', ref.params.L1, 'L2', ref.params.L2);

if isempty(which('rigidBodyTree'))
    fprintf('未安装 Robotics System Toolbox，跳过（不影响主验证）。\n');
    return
end

% ---- 用参考数据中的惯量信息搭建 rigidBodyTree（与 armLab 参数一致）----
robot = rigidBodyTree('DataFormat', 'column', 'MaxNumBodies', 3);
robot.Gravity = ref.params.gravity3(:).';     % 三维重力：[0 -9.81 0]

parent = robot.BaseName;          % 逐个串联：link1 → link2（不能都挂在 base 上）
parent = robot.BaseName;
for i = 1:numel(ref.links)
    L = ref.links(i);
    b = rigidBody(L.name);
    j = rigidBodyJoint(sprintf('q%d', i), 'revolute');
    setFixedTransform(j, trvec2tform(L.pos(:).'));    % 相对父连杆的平移
    j.JointAxis = L.axis(:).';
    b.Joint = j;
    b.Mass = L.mass;
    b.CenterOfMass = L.com(:).';
    I = L.inertia;
    % 注意：MATLAB rigidBody.Inertia 采用 URDF 约定——**绕连杆坐标系原点**表达，
    % 而 armLab 的参数是绕质心，因此这里做平行轴换算。
    com = L.com(:).';
    Io = I + L.mass * (sum(com.^2) * eye(3) - com.' * com);
    b.Inertia = [Io(1,1), Io(2,2), Io(3,3), Io(1,2), Io(1,3), Io(2,3)];
    addBody(robot, b, parent);
    parent = L.name;
end

% ---- 对比 ----
N = min(50, ref.n_states);      % 取前 50 组足够
err_fk = 0; err_M = 0; err_tau = 0;
for i = 1:N
    q = ref.q(i,:)'; dq = ref.dq(i,:)'; ddq = ref.ddq(i,:)';
    T = getTransform(robot, q, 'link2');
    pos_rtb = T(1:2, 4) + T(1:2, 1:2) * [p.L2; 0];   % link2 坐标系 + 末端偏移
    err_fk = max(err_fk, max(abs(pos_rtb - ref.xy(i,:)')));
    err_M  = max(err_M,  max(abs(massMatrix(robot, q) - squeeze(ref.M(i,:,:))), [], 'all'));
    tau_rtb = inverseDynamics(robot, q, dq, ddq);
    err_tau = max(err_tau, max(abs(tau_rtb - ref.tau(i,:)')));
end

fprintf('\n===== P3b Robotics Toolbox(rigidBodyTree) vs armLab =====\n');
fprintf('  样本 %d 组\n', N);
fprintf('  FK 末端位置最大偏差 : %.3e\n', err_fk);
fprintf('  质量矩阵 M 最大偏差 : %.3e\n', err_M);
fprintf('  逆动力学 τ 最大偏差 : %.3e\n', err_tau);
pass = max([err_fk, err_M, err_tau]) < 1e-9;
fprintf('结论: %s（判据 <1e-9）\n\n', string(pass));

out = struct('n_states', N, 'err_fk', err_fk, 'err_M', err_M, 'err_tau', err_tau, ...
             'pass', pass, 'toolbox', 'Robotics System Toolbox rigidBodyTree', ...
             'generated', datestr(now, 'yyyy-mm-dd HH:MM:SS'));
fid = fopen(fullfile(root, 'outputs', 'data', 'matlab_rtb_verification.json'), 'w', 'n', 'UTF-8');
fprintf(fid, '%s', jsonencode(out, 'PrettyPrint', true));
fclose(fid);
fprintf('已写出 outputs/data/matlab_rtb_verification.json\n');
