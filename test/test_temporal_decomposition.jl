using Test
using CrossScaleRC
using Random
using Statistics

@testset "bandpass_decompose" begin
    Random.seed!(0)
    fs    = 1.0
    T     = 1024
    t     = collect(0:T-1) ./ fs

    # Inner window — exclude filtfilt boundary transient (~3*order/cutoff ≈ 72
    # samples for order=4, cutoff=1/24). Use 100-sample margin to be safe.
    margin = 100
    inner  = (margin+1):(T-margin)

    cutoffs = (1/24, 1/3)
    order   = 4

    @testset "Reconstruction (random walk, interior)" begin
        # White noise driven random walk; broadband content stresses all bands.
        sig   = cumsum(randn(T)) ./ sqrt(T)
        bands = bandpass_decompose(sig, cutoffs; fs=fs, order=order)
        recon = reconstruct_bands(bands)
        @test maximum(abs, sig[inner] .- recon[inner]) < 1e-6
    end

    @testset "Energy isolation: pure sine in slow band" begin
        # Period 48 sample = freq 1/48 cyc/sample → in (0, 1/24) → SLOW.
        s     = sin.(2π .* t ./ 48)
        bands = bandpass_decompose(s, cutoffs; fs=fs, order=order)
        @test std(bands.slow[inner]) > 0.6      # ≥ ~85% of signal RMS
        @test std(bands.mid[inner])  < 0.05
        @test std(bands.fast[inner]) < 0.05
    end

    @testset "Energy isolation: pure sine in mid band" begin
        # Period 6 sample = freq 1/6 ∈ (1/24, 1/3) → MID.
        s     = sin.(2π .* t ./ 6)
        bands = bandpass_decompose(s, cutoffs; fs=fs, order=order)
        @test std(bands.mid[inner])  > 0.6
        @test std(bands.slow[inner]) < 0.05
        @test std(bands.fast[inner]) < 0.05
    end

    @testset "Energy isolation: pure sine in fast band" begin
        # Period 1.5 sample = freq 1/1.5 = 0.667 cyc/sample. > Nyquist (0.5)
        # so use period 2.5 → freq 0.4 ∈ (1/3, 1/2) → FAST.
        s     = sin.(2π .* t ./ 2.5)
        bands = bandpass_decompose(s, cutoffs; fs=fs, order=order)
        @test std(bands.fast[inner]) > 0.6
        @test std(bands.slow[inner]) < 0.05
        @test std(bands.mid[inner])  < 0.05
    end

    @testset "Zero-phase property: peak preserved" begin
        # Sharp Gaussian centered at t=512; check slow band's peak coincides
        # (within ±2 samples for finite-order filter).
        c = T ÷ 2
        σ = 80.0   # broad enough for the slow band to capture
        s = exp.(-((t .- c) .^ 2) ./ (2σ^2))
        bands = bandpass_decompose(s, cutoffs; fs=fs, order=order)
        @test abs(argmax(bands.slow[inner]) - argmax(s[inner])) ≤ 2
    end

    @testset "Composite signal: bands separate cleanly" begin
        # Sum of three sines; verify each band picks up its own frequency.
        s_slow = sin.(2π .* t ./ 48)
        s_mid  = sin.(2π .* t ./ 6)
        s_fast = sin.(2π .* t ./ 2.5)
        s = s_slow .+ s_mid .+ s_fast

        bands = bandpass_decompose(s, cutoffs; fs=fs, order=order)
        # Each band should match its target frequency component to within 5%
        # in the interior.
        @test maximum(abs, bands.slow[inner] .- s_slow[inner]) < 0.10
        @test maximum(abs, bands.mid[inner]  .- s_mid[inner])  < 0.10
        @test maximum(abs, bands.fast[inner] .- s_fast[inner]) < 0.10
    end

    @testset "API: 1-cutoff (2-band) form" begin
        sig   = randn(T)
        bands = bandpass_decompose(sig, (0.25,); fs=fs, order=order)
        @test haskey(bands, :slow) && haskey(bands, :fast)
        @test !haskey(bands, :mid)
        recon = reconstruct_bands(bands)
        @test maximum(abs, sig[inner] .- recon[inner]) < 1e-6
    end

    @testset "API: 3-cutoff (4-band) form returns mids vector" begin
        sig   = randn(T)
        bands = bandpass_decompose(sig, (0.05, 0.15, 0.35); fs=fs, order=order)
        @test haskey(bands, :slow)
        @test haskey(bands, :mids) && length(bands.mids) == 2
        @test haskey(bands, :fast)
        recon = reconstruct_bands(bands)
        @test maximum(abs, sig[inner] .- recon[inner]) < 1e-6
    end

    @testset "Validation: rejects bad cutoffs" begin
        sig = randn(T)
        @test_throws AssertionError bandpass_decompose(sig, (1/3, 1/24); fs=fs)  # not sorted
        @test_throws AssertionError bandpass_decompose(sig, (-0.1, 0.3); fs=fs)  # negative
        @test_throws AssertionError bandpass_decompose(sig, (0.3, 0.6); fs=fs)   # > Nyquist
    end
end
