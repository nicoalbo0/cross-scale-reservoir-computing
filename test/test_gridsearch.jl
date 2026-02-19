@testset "grid_search" begin
    # Simple grid: run_once returns a single error
    grids = Dict(:a => [1, 2], :b => [10.0, 20.0])
    outfile = tempname() * ".csv"
    try
        run_once(params) = [params.a * params.b]
        grid_search(run_once, grids; nrep = 2, outfile = outfile, progress = false)
        @test isfile(outfile)
        lines = readlines(outfile)
        @test length(lines) >= 5  # header + 4 grid points
    finally
        isfile(outfile) && rm(outfile)
    end

    # With param_order and error_names
    outfile2 = tempname() * ".csv"
    try
        run_once_2(params) = [params.a * params.b]
        grid_search(run_once_2, grids;
            nrep = 1,
            outfile = outfile2,
            param_order = [:b, :a],
            error_names = [:err1],
            progress = false)
        @test isfile(outfile2)
    finally
        isfile(outfile2) && rm(outfile2)
    end

    # run_once returns multiple errors
    run_twice(params) = [Float64(params.x), 2 * params.x]
    outfile3 = tempname() * ".csv"
    try
        grid_search(run_twice, Dict(:x => [1, 2]); nrep = 2, outfile = outfile3, error_names = [:e1, :e2], progress = false)
        @test isfile(outfile3)
    finally
        isfile(outfile3) && rm(outfile3)
    end

    # Inconsistent length in run_once across reps triggers ArgumentError
    call_count = Ref(0)
    bad_run(params) = (call_count[] += 1; call_count[] == 1 ? [1.0] : [1.0, 2.0])
    @test_throws ArgumentError grid_search(bad_run, Dict(:p => [1, 2]); nrep = 2, outfile = tempname(), progress = false)
end
