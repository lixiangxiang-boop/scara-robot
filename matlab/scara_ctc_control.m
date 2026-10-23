function tau = scara_ctc_control(u)
%SCARA_CTC_CONTROL  Simulink 控制器块：CTC 前馈 + PD + 积分（抗饱和）
%
%   u = [t; qd_ref(4); dqd_ref(4); ddqd_ref(4); q(4); dq(4)]（21×1） → tau(4)
%
%   τ = M(q)[q̈d + Kd(q̇d−q̇) + Kp(qd−q)] + C(q,q̇)q̇ + G(q) + 摩擦前馈(q̇d) + Ki·∫e
%
%   为什么需要积分项：库仑摩擦（本机 J4 的 Fc=0.01 N·m）在低惯量轴上会造成静差
%   e_ss ≈ Fc/(I·Kp)（J4 实测 0.25 rad）；积分项把它压到 ~1e-3 rad 量级。
%   抗饱和：积分项最多补 2 倍库仑摩擦（Ilim = 2·Fc/Ki），与 Python 侧一致。
persistent p Kp Kd Ki Ilim Iacc tlast ready
if isempty(ready)
    p = scara_plant_params();
    % 针对本场景（J4 转 90°）整定：J4 惯量 2e-3 kg·m²，Kp 小会因库仑摩擦
    % 产生静差 e≈Fc/(I·Kp)；Kp4 取 200 后误差从 0.16 rad 降到 0.018 rad
    Kp = [60; 60; 400; 200];
    Kd = [12; 12; 40; 28];
    Ki = Kp / 5;
    Ilim = 2 * p.Fc(:) ./ Ki;
    Iacc = zeros(4, 1);
    tlast = -1;
    ready = true;
end
t = u(1);
qd_ref   = u(2:5);
dqd_ref  = u(6:9);
ddqd_ref = u(10:13);
q  = u(14:17);
dq = u(18:21);
dt = 0;
if tlast >= 0
    dt = t - tlast;
    if ~isfinite(dt) || dt < 0 || dt > 0.05, dt = 0; end      % 步长异常/回退时不计分
end
tlast = t;
e  = qd_ref - q;
de = dqd_ref - dq;
Iacc = Iacc + e * dt;
Iacc = max(min(Iacc, Ilim), -Ilim);
[M, Cqd, G, ~] = scara_plant_dynamics(q, dq, p, 0.02);
Fff = p.Fv(:).*dqd_ref + p.Fc(:).*tanh(dqd_ref/0.02);
tau = M*(ddqd_ref + Kd.*de + Kp.*e) + Cqd + G + Fff + Ki.*Iacc;
end
