namespace MachineLearning.Tests

open MachineLearning
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

    [<Test>]
    member this.``Majority should return most common element in last column`` () =
        let dataset =
            [| "A"; "B"; "Class" |],
            [| [| "Yes"; "Yes"; "Green" |];
               [| "Yes"; "Yes"; "Green" |];
               [| "Yes"; "No";  "Red"  |]; |]
        let expected = "Green"
        let actual = majority dataset

        actual |> should equal expected

    [<Test>]
    member this.``Classify should match subject with tree`` () =
        let tree =
            Choice("First", 
                [| "Head", Choice("Second", 
                    [| "Head", Conclusion("Win"); 
                       "Tail", Conclusion("Lose") |]); 
                    "Tail", Conclusion("Lose") |])

        let subject = [("First", "Head"); ("Second", "Head")]
        let expected = "Win"
        let actual = classify subject tree

        actual |> should equal expected

        let subject = [("First", "Tail")]
        let expected = "Lose"
        let actual = classify subject tree

        actual |> should equal expected

        let subject = [("First", "Head"); ("Second", "Tail")]
        let expected = "Lose"
        let actual = classify subject tree

        actual |> should equal expected