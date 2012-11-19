namespace MachineLearning.Tests

open MachineLearning.SupportVectorMachine
open NUnit.Framework
open FsUnit

[<TestFixture>]
type ``SVM tests`` () =

    [<Test>]
    member this.``dot verification`` () =

        let v1 = [ 1.0; 2.0 ]
        let v2 = [ 3.0; 4.0 ]

        dot v1 v2 |> should equal 11.0

    [<Test>]
    member this.``clip verification`` () =

        let min, max = 1.0, 3.0

        clip (min, max) 0.0 |> should equal min
        clip (min, max) 10.0 |> should equal max
        clip (min, max) min |> should equal min
        clip (min, max) max |> should equal max
        clip (min, max) 2.0 |> should equal 2.0

    [<Test>]
    member this.``nextAround verification`` () =
        let size = 5
        nextAround size 0 |> should equal 1
        nextAround size 4 |> should equal 0