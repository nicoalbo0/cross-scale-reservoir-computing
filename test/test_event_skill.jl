using Test, Random, Statistics, CrossScaleRC

@testset "find_enso_events" begin
    # Synthetic series with three known peaks
    nt = 96
    n34 = zeros(Float64, nt)
    n34[20] = +2.0     # warm event
    n34[50] = -2.5     # strong cold
    n34[80] = -1.5     # cold event

    events = find_enso_events(n34; threshold = 1.0, min_separation = 6, n_events = 6)
    @test events == [20, 50, 80]

    # Threshold cuts the weakest
    events_strict = find_enso_events(n34; threshold = 1.6, n_events = 6)
    @test events_strict == [20, 50]

    # Subthreshold series → empty
    events_empty = find_enso_events(0.5 .* sin.(1:nt); threshold = 1.0, n_events = 6)
    @test isempty(events_empty)
end

@testset "phase_aligned_pc — perfect prediction" begin
    nlon, nlat, nt = 10, 8, 30
    Random.seed!(0)
    field_true = randn(nlon, nlat, nt)
    field_pred = copy(field_true)

    # Truth at t=15 should match itself with pc=1, offset=0
    r = phase_aligned_pc(field_true, field_pred, 15; phase_tol = 3)
    @test isapprox(r.best_pc, 1.0; atol = 1e-9)
    @test r.best_offset == 0
end

@testset "phase_aligned_pc — phase shift within tolerance" begin
    nlon, nlat, nt = 10, 8, 30
    Random.seed!(1)
    field_true = randn(nlon, nlat, nt)
    # Build field_pred = field_true shifted by +2 mo (forecast leads truth by 2)
    field_pred = zeros(nlon, nlat, nt)
    for t in 1:nt
        src = t - 2  # pred at month t = truth at month t-2 → pred LEADS truth by 2
        if 1 ≤ src ≤ nt
            field_pred[:, :, t] .= field_true[:, :, src]
        end
    end
    # Truth at t=15: in pred this pattern appears at t=17 (= 15 + 2)
    r = phase_aligned_pc(field_true, field_pred, 15; phase_tol = 3)
    @test isapprox(r.best_pc, 1.0; atol = 1e-9)
    @test r.best_offset == 2
end

@testset "phase_aligned_pc — shift outside tolerance" begin
    nlon, nlat, nt = 10, 8, 30
    Random.seed!(2)
    field_true = randn(nlon, nlat, nt)
    field_pred = zeros(nlon, nlat, nt)
    # Shift by +6 — outside ±3 tolerance
    for t in 1:nt
        src = t - 6
        if 1 ≤ src ≤ nt
            field_pred[:, :, t] .= field_true[:, :, src]
        end
    end
    r = phase_aligned_pc(field_true, field_pred, 15; phase_tol = 3)
    @test r.best_pc < 0.5    # shouldn't find a high pc within ±3
end

@testset "event_skill — perfect forecast" begin
    nlon, nlat, nt = 10, 8, 96
    Random.seed!(10)
    n34_true = zeros(nt)
    n34_true[15] = +2.0
    n34_true[40] = -2.5

    # Field: a per-event spatial pattern injected at event times
    field_true = 0.05 .* randn(nlon, nlat, nt)
    pattern_warm = randn(nlon, nlat)
    pattern_cold = -randn(nlon, nlat)
    field_true[:, :, 15] .+= pattern_warm
    field_true[:, :, 40] .+= pattern_cold

    n34_pred   = copy(n34_true)
    field_pred = copy(field_true)

    s = event_skill(n34_true, field_true, n34_pred, field_pred;
                    lead_window = (10, 50), phase_tol = 3,
                    event_threshold = 1.0, min_separation = 6)
    @test s.n_events == 2
    @test s.events_true == [15, 40]
    @test s.sign_accuracy == 1.0
    @test s.mean_event_pc > 0.99
    @test s.false_alarms == 0
end

@testset "event_skill — phase-shifted forecast within tolerance" begin
    nlon, nlat, nt = 10, 8, 96
    Random.seed!(11)
    n34_true = zeros(nt); n34_true[15] = +2.0; n34_true[40] = -2.5
    field_true = 0.05 .* randn(nlon, nlat, nt)
    pattern_warm = randn(nlon, nlat); pattern_cold = -randn(nlon, nlat)
    field_true[:, :, 15] .+= pattern_warm
    field_true[:, :, 40] .+= pattern_cold

    # Forecast leads truth by 2 mo
    n34_pred = zeros(nt); n34_pred[13] = +2.0; n34_pred[38] = -2.5
    field_pred = 0.05 .* randn(nlon, nlat, nt)
    field_pred[:, :, 13] .+= pattern_warm
    field_pred[:, :, 38] .+= pattern_cold

    s = event_skill(n34_true, field_true, n34_pred, field_pred;
                    lead_window = (10, 50), phase_tol = 3)
    @test s.sign_accuracy == 1.0
    @test s.mean_event_pc > 0.9
    @test all(s.best_offset .== -2)  # truth_event + δ = pred_event ⇒ δ = -2
end

@testset "event_skill — phase shift outside tolerance" begin
    nlon, nlat, nt = 10, 8, 96
    Random.seed!(12)
    n34_true = zeros(nt); n34_true[15] = +2.0; n34_true[40] = -2.5
    field_true = 0.05 .* randn(nlon, nlat, nt)
    pattern_warm = randn(nlon, nlat); pattern_cold = -randn(nlon, nlat)
    field_true[:, :, 15] .+= pattern_warm
    field_true[:, :, 40] .+= pattern_cold

    # Forecast leads by 6 mo — outside ±3 tolerance
    n34_pred = zeros(nt); n34_pred[9] = +2.0; n34_pred[34] = -2.5
    field_pred = 0.05 .* randn(nlon, nlat, nt)
    field_pred[:, :, 9] .+= pattern_warm
    field_pred[:, :, 34] .+= pattern_cold

    s = event_skill(n34_true, field_true, n34_pred, field_pred;
                    lead_window = (10, 50), phase_tol = 3)
    @test s.mean_event_pc < 0.5     # patterns don't align within ±3
    # Forecast peaks at L=9 and L=34 are outside the lead_window=(10,50);
    # only L=34 is in window. It's not within phase_tol of any truth event
    # (closest is t=40, |34-40|=6>3) → false alarm.
    @test s.false_alarms ≥ 1
end

@testset "event_skill — silent forecast (no false alarms, no skill)" begin
    nlon, nlat, nt = 10, 8, 96
    Random.seed!(13)
    n34_true = zeros(nt); n34_true[15] = +2.0; n34_true[40] = -2.5
    field_true = 0.05 .* randn(nlon, nlat, nt)
    field_true[:, :, 15] .+= randn(nlon, nlat)
    field_true[:, :, 40] .-= randn(nlon, nlat)

    n34_pred   = zeros(nt)
    field_pred = zeros(nlon, nlat, nt)

    s = event_skill(n34_true, field_true, n34_pred, field_pred;
                    lead_window = (10, 50), phase_tol = 3)
    @test s.false_alarms == 0
    @test s.sign_accuracy == 0.0       # zero forecast has sign 0, never matches ±
    # mean_event_pc should be ~0 (zero pred has zero variance → pc near 0)
    @test abs(s.mean_event_pc) < 0.1
end

@testset "event_skill — events outside lead_window are excluded" begin
    nlon, nlat, nt = 10, 8, 96
    n34_true = zeros(nt); n34_true[5] = +2.0; n34_true[15] = +1.5; n34_true[80] = -2.0
    field_true = randn(nlon, nlat, nt)
    field_pred = copy(field_true)

    s = event_skill(n34_true, field_true, n34_true, field_pred;
                    lead_window = (10, 50), phase_tol = 3)
    @test s.n_events == 1     # only t=15 lies inside (10, 50)
    @test s.events_true == [15]
end
