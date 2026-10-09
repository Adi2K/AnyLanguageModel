import Testing

@testable import AnyLanguageModel

#if MLX
    import enum MLXLMCommon.Chat
    import struct MLXLMCommon.DefaultMessageGenerator
    import struct MLXLMCommon.ToolCall

    @Suite("MLX tool round chat")
    struct MLXToolRoundTests {
        private typealias RawMessage = [String: any Sendable]

        /// A call and the result that must directly follow it.
        private struct Exchange {
            let name: String
            let id: String?
            let result: String
        }

        private func call(_ name: String, id: String? = nil) -> ToolCall {
            ToolCall(function: .init(name: name, arguments: ["city": .string("Paris")]), id: id)
        }

        /// A call whose arguments hold a null next to the usual city.
        private func callWithNullArgument(_ name: String, id: String? = nil) -> ToolCall {
            ToolCall(function: .init(name: name, arguments: ["city": .string("Paris"), "unit": .null]), id: id)
        }

        private func raw(_ chat: [Chat.Message]) -> [RawMessage] {
            DefaultMessageGenerator().generate(messages: chat)
        }

        /// Checks that `messages` holds, from `start`, one assistant message per exchange,
        /// each carrying exactly that call and directly followed by its tool message.
        private func expectExchanges(
            _ exchanges: [Exchange],
            in messages: [RawMessage],
            from start: Int = 0
        ) throws {
            try #require(messages.count >= start + exchanges.count * 2)
            for (offset, exchange) in exchanges.enumerated() {
                let assistant = messages[start + offset * 2]
                let tool = messages[start + offset * 2 + 1]

                #expect(assistant["role"] as? String == "assistant")
                #expect(assistant["content"] as? String == "")
                let calls = try #require(assistant["tool_calls"] as? [[String: any Sendable]])
                #expect(calls.count == 1)
                let function = try #require(calls.first?["function"] as? [String: any Sendable])
                #expect(function["name"] as? String == exchange.name)
                let arguments = try #require(function["arguments"] as? [String: any Sendable])
                #expect(arguments["city"] as? String == "Paris")
                #expect(arguments.keys.sorted() == ["city"])

                #expect(tool["role"] as? String == "tool")
                #expect(tool["content"] as? String == exchange.result)
                #expect(tool["name"] as? String == exchange.name)

                if let id = exchange.id {
                    #expect(calls.first?["id"] as? String == id)
                    #expect(tool["tool_call_id"] as? String == id)
                } else {
                    #expect(calls.first?["id"] == nil)
                    #expect(tool["tool_call_id"] == nil)
                }
            }
        }

        private func converted(_ entries: [Transcript.Entry]) -> [RawMessage] {
            let session = LanguageModelSession(
                model: MockLanguageModel.fixed("unused"),
                transcript: Transcript(entries: entries)
            )
            return raw(
                convertTranscriptToMLXChat(
                    requestContext: session.resolvedRequestContext(),
                    fallbackPrompt: "unused"
                )
            )
        }

        private func roles(_ messages: [RawMessage]) -> [String?] {
            messages.map { $0["role"] as? String }
        }

        private func prompt(_ text: String) -> Transcript.Entry {
            .prompt(Transcript.Prompt(segments: [.text(.init(content: text))]))
        }

        private func response(_ text: String) -> Transcript.Entry {
            .response(Transcript.Response(assetIDs: [], segments: [.text(.init(content: text))]))
        }

        private func transcriptCall(
            _ name: String,
            id: String,
            arguments: String = #"{"city":"Paris"}"#
        ) throws -> Transcript.ToolCall {
            try Transcript.ToolCall(id: id, toolName: name, arguments: GeneratedContent(json: arguments))
        }

        private func output(_ text: String, id: String, toolName: String) -> Transcript.Entry {
            .toolOutput(Transcript.ToolOutput(id: id, toolName: toolName, segments: [.text(.init(content: text))]))
        }

        @Test func toolRoundPutsTheCallAheadOfItsResult() throws {
            let messages = raw(
                makeMLXToolRoundMessages(
                    calls: [call("get_weather", id: "call_1")],
                    results: ["sunny"]
                )
            )

            #expect(messages.count == 2)
            try expectExchanges([Exchange(name: "get_weather", id: "call_1", result: "sunny")], in: messages)
        }

        @Test func toolRoundLeavesANullArgumentOutOfTheReplayedCall() throws {
            let messages = raw(
                makeMLXToolRoundMessages(
                    calls: [callWithNullArgument("get_weather", id: "call_1")],
                    results: ["sunny"]
                )
            )

            #expect(messages.count == 2)
            try expectExchanges([Exchange(name: "get_weather", id: "call_1", result: "sunny")], in: messages)
        }

        @Test func toolRoundLeavesNestedNullsOutOfTheReplayedCall() throws {
            let nested = ToolCall(
                function: .init(
                    name: "get_weather",
                    arguments: [
                        "city": .string("Paris"),
                        "filter": .object(["unit": .null, "days": .int(3)]),
                        "tags": .array([.string("a"), .null]),
                    ]
                ),
                id: "call_1"
            )
            let messages = raw(makeMLXToolRoundMessages(calls: [nested], results: ["sunny"]))

            let calls = try #require(messages.first?["tool_calls"] as? [[String: any Sendable]])
            let function = try #require(calls.first?["function"] as? [String: any Sendable])
            let arguments = try #require(function["arguments"] as? [String: any Sendable])
            #expect(arguments["city"] as? String == "Paris")
            let filter = try #require(arguments["filter"] as? [String: any Sendable])
            #expect(filter.keys.sorted() == ["days"])
            let tags = try #require(arguments["tags"] as? [any Sendable])
            #expect(tags.count == 1)
            #expect(tags.first as? String == "a")
        }

        @Test func toolRoundWithTwoCallsGivesEachCallItsOwnMessage() throws {
            let messages = raw(
                makeMLXToolRoundMessages(
                    calls: [call("get_weather", id: "call_1"), call("get_time", id: "call_2")],
                    results: ["sunny", "14:05"]
                )
            )

            #expect(messages.count == 4)
            try expectExchanges(
                [
                    Exchange(name: "get_weather", id: "call_1", result: "sunny"),
                    Exchange(name: "get_time", id: "call_2", result: "14:05"),
                ],
                in: messages
            )
        }

        @Test func toolRoundWithoutCallIdsStillNamesEachResult() throws {
            let messages = raw(
                makeMLXToolRoundMessages(
                    calls: [call("get_weather"), call("get_time")],
                    results: ["sunny", "14:05"]
                )
            )

            #expect(messages.count == 4)
            try expectExchanges(
                [
                    Exchange(name: "get_weather", id: nil, result: "sunny"),
                    Exchange(name: "get_time", id: nil, result: "14:05"),
                ],
                in: messages
            )
        }

        @Test func transcriptToolRoundIsReplayedCallThenResult() throws {
            let messages = try converted([
                prompt("What is the weather in Paris?"),
                .toolCalls(Transcript.ToolCalls([transcriptCall("get_weather", id: "id-1")])),
                output("sunny", id: "id-1", toolName: "get_weather"),
                response("It is sunny."),
            ])

            #expect(roles(messages) == ["user", "assistant", "tool", "assistant"])
            try expectExchanges([Exchange(name: "get_weather", id: "id-1", result: "sunny")], in: messages, from: 1)
            #expect(messages.last?["content"] as? String == "It is sunny.")
            #expect(messages.last?["tool_calls"] == nil)
        }

        @Test func transcriptToolOutputsArePairedById() throws {
            let messages = try converted([
                prompt("What is the weather and the time in Paris?"),
                .toolCalls(
                    Transcript.ToolCalls([
                        transcriptCall("get_weather", id: "id-1"),
                        transcriptCall("get_time", id: "id-2"),
                    ])
                ),
                output("14:05", id: "id-2", toolName: "get_time"),
                output("sunny", id: "id-1", toolName: "get_weather"),
            ])

            #expect(messages.count == 5)
            try expectExchanges(
                [
                    Exchange(name: "get_time", id: "id-2", result: "14:05"),
                    Exchange(name: "get_weather", id: "id-1", result: "sunny"),
                ],
                in: messages,
                from: 1
            )
        }

        @Test func transcriptToolOutputsWithOtherIdsArePairedInCallOrder() throws {
            let messages = try converted([
                prompt("What is the weather and the time in Paris?"),
                .toolCalls(
                    Transcript.ToolCalls([
                        transcriptCall("get_weather", id: "id-1"),
                        transcriptCall("get_time", id: "id-2"),
                    ])
                ),
                output("sunny", id: "other-1", toolName: "get_weather"),
                output("14:05", id: "other-2", toolName: "get_time"),
            ])

            #expect(messages.count == 5)
            try expectExchanges(
                [
                    Exchange(name: "get_weather", id: "id-1", result: "sunny"),
                    Exchange(name: "get_time", id: "id-2", result: "14:05"),
                ],
                in: messages,
                from: 1
            )
        }

        @Test func transcriptToolCallsWithoutOutputsAreNotReplayed() throws {
            let messages = try converted([
                prompt("What is the weather in Paris?"),
                .toolCalls(Transcript.ToolCalls([transcriptCall("get_weather", id: "id-1")])),
                response(""),
            ])

            #expect(roles(messages) == ["user", "assistant"])
            #expect(messages.allSatisfy { $0["tool_calls"] == nil })
        }

        @Test func transcriptToolOutputWithoutCallKeepsItsIdAndToolName() {
            let messages = converted([
                prompt("What is the weather in Paris?"),
                output("sunny", id: "id-1", toolName: "get_weather"),
            ])

            #expect(roles(messages) == ["user", "tool"])
            #expect(messages.last?["content"] as? String == "sunny")
            #expect(messages.last?["tool_call_id"] as? String == "id-1")
            #expect(messages.last?["name"] as? String == "get_weather")
        }

        @Test func transcriptToolCallWithANullArgumentIsReplayedWithoutIt() throws {
            let messages = try converted([
                prompt("What is the weather in Paris?"),
                .toolCalls(
                    Transcript.ToolCalls([
                        transcriptCall("get_weather", id: "id-1", arguments: #"{"city":"Paris","unit":null}"#)
                    ])
                ),
                output("sunny", id: "id-1", toolName: "get_weather"),
            ])

            #expect(roles(messages) == ["user", "assistant", "tool"])
            try expectExchanges([Exchange(name: "get_weather", id: "id-1", result: "sunny")], in: messages, from: 1)
        }

        @Test func transcriptToolCallsFromAnEarlierStoppedRoundAreNotPairedWithLaterOutputs() throws {
            let messages = try converted([
                prompt("What is the weather in Paris?"),
                .toolCalls(Transcript.ToolCalls([transcriptCall("get_weather", id: "id-1")])),
                response(""),
                prompt("What time is it in Paris?"),
                .toolCalls(Transcript.ToolCalls([transcriptCall("get_time", id: "id-2")])),
                output("14:05", id: "other", toolName: "get_time"),
            ])

            #expect(roles(messages) == ["user", "assistant", "user", "assistant", "tool"])
            try expectExchanges([Exchange(name: "get_time", id: "id-2", result: "14:05")], in: messages, from: 3)
        }

        @Test func transcriptToolOutputAfterALaterPromptIsNotPairedWithAnEarlierCall() throws {
            let messages = try converted([
                prompt("What is the weather in Paris?"),
                .toolCalls(Transcript.ToolCalls([transcriptCall("get_weather", id: "id-1")])),
                prompt("What time is it in Paris?"),
                output("14:05", id: "other", toolName: "get_time"),
            ])

            #expect(roles(messages) == ["user", "user", "tool"])
            #expect(messages.allSatisfy { $0["tool_calls"] == nil })
            #expect(messages.last?["content"] as? String == "14:05")
        }

        @Test func transcriptToolOutputAfterAResponseIsNotPairedWithAnEarlierCall() throws {
            let messages = try converted([
                prompt("What is the weather in Paris?"),
                .toolCalls(Transcript.ToolCalls([transcriptCall("get_weather", id: "id-1")])),
                response(""),
                output("14:05", id: "other", toolName: "get_time"),
            ])

            #expect(roles(messages) == ["user", "assistant", "tool"])
            #expect(messages.allSatisfy { $0["tool_calls"] == nil })
            #expect(messages.last?["content"] as? String == "14:05")
        }

        @Test func transcriptToolOutputAfterAResponseIsNotPairedWithAnEarlierCallOfTheSameId() throws {
            let messages = try converted([
                prompt("What is the weather in Paris?"),
                .toolCalls(Transcript.ToolCalls([transcriptCall("get_weather", id: "id-1")])),
                response("Let me check."),
                output("sunny", id: "id-1", toolName: "get_weather"),
            ])

            #expect(roles(messages) == ["user", "assistant", "tool"])
            #expect(messages.allSatisfy { $0["tool_calls"] == nil })
            #expect(messages.last?["content"] as? String == "sunny")
        }

        @Test func transcriptWithTwoToolRoundsReplaysEachRoundWithItsOwnOutput() throws {
            let messages = try converted([
                prompt("What is the weather and the time in Paris?"),
                .toolCalls(Transcript.ToolCalls([transcriptCall("get_weather", id: "id-1")])),
                output("sunny", id: "id-1", toolName: "get_weather"),
                .toolCalls(Transcript.ToolCalls([transcriptCall("get_time", id: "id-2")])),
                output("14:05", id: "id-2", toolName: "get_time"),
                response("Sunny, and it is 14:05."),
            ])

            #expect(roles(messages) == ["user", "assistant", "tool", "assistant", "tool", "assistant"])
            try expectExchanges(
                [
                    Exchange(name: "get_weather", id: "id-1", result: "sunny"),
                    Exchange(name: "get_time", id: "id-2", result: "14:05"),
                ],
                in: messages,
                from: 1
            )
            #expect(messages.last?["content"] as? String == "Sunny, and it is 14:05.")
            #expect(messages.last?["tool_calls"] == nil)
        }
    }
#endif  // MLX
