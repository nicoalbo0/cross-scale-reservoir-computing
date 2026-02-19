@testset "loading" begin
    using CSV, DataFrames

    # load_data(::Int, ::Int, ::Real) dispatches to load_kuramoto_data
    # Create minimal KS data file in temp dir and run load_kuramoto_data path
    let
        orig_pwd = pwd()
        tmp = mktempdir()
        try
            cd(tmp)
            mkpath("data/kuramoto")
            # KS file: Q, L, μ in filename. load_kuramoto_data expects CSV -> Matrix' -> (L × T)
            # So CSV with L rows and T columns -> Matrix (L, T), transpose -> (T, L); then mean(..., dims=2) is over columns -> (T,) so we get (T,L). Actually in Julia size(Matrix(DataFrame)) is (nrow, ncol). So (L, T) if we write L rows, T cols. Then f = (...)' gives (T, L). Then data = (f .- mean(f,dims=2)) ./ std(f,dims=2): dims=2 means along dimension 2, so for (T,L) we normalize each row (each of T rows). So result is (T, L). So data is (T, L) but doc says "data is (L×T)". So either doc is wrong or the code. Looking again: "Returns (data, dt) where data is (L×T)". So L is first dimension. So we need data (L, T). So f should be (L, T). So Matrix(DataFrame)' = (L, T) means Matrix(DataFrame) is (T, L), so DataFrame has T rows and L columns. So write CSV with T rows, L columns.
            Q, L, μ = 2, 3, 0.01
            T_csv = 20
            mat = randn(L, T_csv)
            df = DataFrame(mat', :auto)
            CSV.write("data/kuramoto/Q$(Q)_L$(L)_mu$(μ)_ks_data.csv", df)
            data, dt = load_data(Q, L, μ; show_data=false, refinement=1)
            @test size(data, 1) == L
            @test size(data, 2) == T_csv
            @test dt == 0.25
            # refinement > 1 branch
            data2, dt2 = load_data(Q, L, μ; show_data=false, refinement=2)
            @test size(data2, 1) == L
            @test size(data2, 2) > T_csv
            @test dt2 ≈ 0.25 / 2
        finally
            cd(orig_pwd)
        end
    end

    # load_data(::Vector{Float64}) dispatches to load_sst_data - requires JLD2 files; skip if no data
    # We only test that the method is callable and errors informatively when files missing
    @test_throws Exception load_data([99.0]; show_data=false, refinement=1)
end
