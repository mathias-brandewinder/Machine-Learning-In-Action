namespace AdaBoost.Tests

open MachineLearning.AdaBoost
open NUnit.Framework
open FsUnit

[<TestFixture>]
type ``AdaBoost tests`` () =

    [<Test>]
    member this.``normalize verification`` () =

        let v = [| 1.0; 3.0; 4.0 |]

        normalize v |> should equal [| 0.125; 0.375; 0.500|]
