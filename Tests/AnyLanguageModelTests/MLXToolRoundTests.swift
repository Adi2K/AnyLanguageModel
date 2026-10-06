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
    }
#endif  // MLX
