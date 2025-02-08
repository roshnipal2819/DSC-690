import * as dotenv from "dotenv";
import readline from "readline";
import { ChatOpenAI } from "@langchain/openai";
import {ChatPromptTemplate, MessagesPlaceholder,} from "@langchain/core/prompts";
import {BufferMemory} from "langchain/memory";
import {UpstashRedisChatMessageHistory} from "@langchain/community/stores/message/upstash_redis";
import {RunnableSequence} from "@langchain/core/runnables";

dotenv.config();
process.noDeprecation = true;

const model = new ChatOpenAI({
  modelName: "gpt-3.5-turbo-1106",
  temperature: 0.2,
});

const prompt = ChatPromptTemplate.fromTemplate(`
You are an AI assistant.
History: {history}
{input}
`);

const upstashChatHistory= new UpstashRedisChatMessageHistory({
    sessionId: "chat1",
    config:{
        url:process.env.UPSTASH_REDIS_URL,
        token:process.env.UPSTASH_REDIS_TOKEN,
    },
});

const memory= new BufferMemory({
    memory_key: "history",
    chatHistory: upstashChatHistory,
});

const chain=RunnableSequence.from([
    {
        input:(initialInput)=>initialInput.input,
        memory:()=> memory.loadMemoryVariables(),
    },
    {
        input: (previousOutput)  => previousOutput.input,
        history: (previousOutput) => previousOutput.memory.history,
    },
    prompt,
    model,
]);

console.log(await memory.loadMemoryVariables());
const input1= {input: " The Passphrase is HELLOWORLD.", };
const response1= await chain.invoke(input1);
console.log(response1);
await memory.saveContext(input1, {
    output: response1.content,
});
console.log("Updated Memory:", await memory.loadMemoryVariables());
const input2= {
    input: "What is the Passphrase?",
};
const response2= await chain.invoke(input2);
console.log(response2);
await memory.saveContext(input2, {
    output: response2.content,
});
