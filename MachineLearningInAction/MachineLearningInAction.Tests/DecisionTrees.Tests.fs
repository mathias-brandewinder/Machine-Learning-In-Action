namespace MachineLearning.Tests

open MachineLearning.DecisionTrees
open NUnit.Framework
open FsUnit

[<TestFixture>]
type ``Decision Trees tests`` () =

    [<Test>]
    member this.``Remove vector i should remove ith component`` () =
        let vector = [| "A"; "B"; "C" |]
        let expected = [| "A"; "C" |]
        let actual = remove 1 vector

        actual |> should equal expected
