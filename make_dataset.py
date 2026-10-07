#!/usr/bin/env python3
"""
Build a clean, diverse training dataset for the engineering model.
Format: ### User: / ### Assistant: — matches Modelfile.finetuned template.
No podcast transcripts or filler speech data.
"""

import json
import random
from pathlib import Path

# --- High-quality Q&A pairs ---
# Style: direct, technical, teaches concepts with code examples.
# This is the style you want the model to output.

EXAMPLES = [
    # JavaScript / Array algorithms
    {
        "text": "### User:\nTeach me step by step how to flatten an array in JavaScript\n### Assistant:\nFlattening an array means turning nested arrays into a single flat array.\n\n**Step 1 — The built-in way (modern JS):**\n```javascript\nconst nested = [1, [2, 3], [4, [5, 6]]];\n\n// Flatten one level deep\nconst flat1 = nested.flat();\n// [1, 2, 3, 4, [5, 6]]\n\n// Flatten all levels\nconst flatAll = nested.flat(Infinity);\n// [1, 2, 3, 4, 5, 6]\n```\n\n**Step 2 — How it works internally (recursive approach):**\n```javascript\nconst flattenArray = (arr) =>\n  arr.reduce((acc, item) =>\n    Array.isArray(item)\n      ? acc.concat(flattenArray(item))\n      : acc.concat(item),\n  []);\n\nconsole.log(flattenArray([1, [2, [3, [4]]]])); // [1, 2, 3, 4]\n```\n\n**Step 3 — Understanding `reduce` here:**\n- `acc` starts as `[]`\n- For each `item`: if it's an array, recurse; otherwise push it to `acc`\n- Returns the accumulated flat array\n\n**When to use which:**\n- `.flat(Infinity)` — simple, readable, use this by default\n- Recursive version — when you need custom logic (e.g. filter while flattening)"
    },
    {
        "text": "### User:\nExplain JavaScript closures with a real-world example\n### Assistant:\nA closure is a function that remembers the variables from its outer scope even after that scope has finished executing.\n\n**Real-world example — a counter factory:**\n```javascript\nconst makeCounter = (start = 0) => {\n  let count = start; // this variable is 'closed over'\n\n  return {\n    increment: () => ++count,\n    decrement: () => --count,\n    value: () => count,\n  };\n};\n\nconst counter = makeCounter(10);\nconsole.log(counter.increment()); // 11\nconsole.log(counter.increment()); // 12\nconsole.log(counter.decrement()); // 11\nconsole.log(counter.value());     // 11\n```\n\n**Why this is a closure:**\n- `makeCounter` runs and finishes\n- But `increment`, `decrement`, and `value` still have access to `count`\n- Each call to `makeCounter()` creates an independent `count` variable\n\n**Common use cases:**\n- Private state (like above)\n- Event handlers that remember data\n- Memoization / caching"
    },
    {
        "text": "### User:\nHow does the event loop work in JavaScript? Explain step by step.\n### Assistant:\nJavaScript is single-threaded — it can only do one thing at a time. The event loop is how it handles async operations without blocking.\n\n**The pieces:**\n1. **Call Stack** — where your code executes, one frame at a time\n2. **Web APIs** — browser handles timers, fetch, events (not JS itself)\n3. **Task Queue** — completed callbacks wait here\n4. **Microtask Queue** — Promise `.then()` callbacks live here (higher priority)\n\n**Step by step example:**\n```javascript\nconsole.log('1');                        // call stack\n\nsetTimeout(() => console.log('2'), 0);  // handed to Web API\n\nPromise.resolve().then(() => console.log('3')); // microtask queue\n\nconsole.log('4');                        // call stack\n\n// Output: 1, 4, 3, 2\n```\n\n**Why that order?**\n- `1` and `4` run synchronously (call stack)\n- After stack is empty: microtasks run first → `3`\n- Then task queue → `2`\n\n**Key rule:** Microtasks (Promises) always flush before the next task queue item."
    },
    {
        "text": "### User:\nWhat is the difference between `==` and `===` in JavaScript?\n### Assistant:\n`==` is loose equality — it converts types before comparing.\n`===` is strict equality — it requires both value AND type to match.\n\n**Examples:**\n```javascript\n// Loose equality (==) — coerces types\nconsole.log(0 == false);   // true  (false → 0)\nconsole.log('' == false);  // true  (both → 0)\nconsole.log(null == undefined); // true (special case)\nconsole.log('5' == 5);    // true  ('5' → 5)\n\n// Strict equality (===) — no coercion\nconsole.log(0 === false);  // false (number vs boolean)\nconsole.log('5' === 5);   // false (string vs number)\nconsole.log(null === undefined); // false (different types)\n```\n\n**Rule of thumb:** Always use `===`. Using `==` requires you to memorize JavaScript's type coercion table, which is a source of bugs.\n\nThe only common exception: `x == null` checks for both `null` and `undefined` in one expression, which some codebases use intentionally."
    },
    {
        "text": "### User:\nWrite a function to debounce in JavaScript and explain how it works\n### Assistant:\nDebouncing delays a function call until after a user stops triggering it.\n\n**Implementation:**\n```javascript\nconst debounce = (fn, delay) => {\n  let timer;\n  return (...args) => {\n    clearTimeout(timer);\n    timer = setTimeout(() => fn(...args), delay);\n  };\n};\n\n// Usage: only fires 300ms after the user stops typing\nconst handleSearch = debounce((query) => {\n  console.log('Searching for:', query);\n}, 300);\n\ninput.addEventListener('input', (e) => handleSearch(e.target.value));\n```\n\n**How it works step by step:**\n1. User types → `handleSearch` is called\n2. Any existing timer is cleared (`clearTimeout`)\n3. A new 300ms timer starts\n4. If user types again within 300ms → go back to step 2\n5. If 300ms passes without a new call → `fn` finally executes\n\n**Real-world uses:** Search inputs, window resize handlers, form auto-save."
    },
    {
        "text": "### User:\nExplain promises in JavaScript — what problem do they solve?\n### Assistant:\nPromises solve callback hell — deeply nested async code that's hard to read and handle errors in.\n\n**The problem (callback hell):**\n```javascript\ngetUser(id, (user) => {\n  getOrders(user.id, (orders) => {\n    getProduct(orders[0].productId, (product) => {\n      // buried 3 levels deep, error handling is a mess\n    });\n  });\n});\n```\n\n**The solution (Promise chain):**\n```javascript\ngetUser(id)\n  .then(user => getOrders(user.id))\n  .then(orders => getProduct(orders[0].productId))\n  .then(product => console.log(product))\n  .catch(err => console.error('Something failed:', err));\n```\n\n**A Promise has 3 states:**\n- `pending` — in progress\n- `fulfilled` — completed with a value\n- `rejected` — failed with an error\n\n**Creating one:**\n```javascript\nconst wait = (ms) => new Promise((resolve) => setTimeout(resolve, ms));\n\nawait wait(1000); // pauses for 1 second\n```\n\n**Today, use async/await** — it's built on Promises but reads like synchronous code."
    },
    # TypeScript
    {
        "text": "### User:\nWhat are TypeScript generics and when should I use them?\n### Assistant:\nGenerics let you write reusable code that works with multiple types while staying type-safe.\n\n**Without generics — you'd need duplicates:**\n```typescript\nconst firstNumber = (arr: number[]): number => arr[0];\nconst firstString = (arr: string[]): string => arr[0];\n```\n\n**With generics — one function:**\n```typescript\nconst first = <T>(arr: T[]): T => arr[0];\n\nfirst([1, 2, 3]);       // returns number\nfirst(['a', 'b', 'c']); // returns string\n```\n\n**Constrained generics:**\n```typescript\n// T must have a .length property\nconst longest = <T extends { length: number }>(a: T, b: T): T =>\n  a.length >= b.length ? a : b;\n\nlongest('hello', 'hi');   // 'hello'\nlongest([1, 2], [3]);     // [1, 2]\n```\n\n**When to use generics:**\n- Utility functions (first, last, groupBy)\n- API response wrappers: `ApiResponse<T>`\n- Reusable hooks: `useLocalStorage<T>(key, defaultValue)`\n- Anywhere you'd otherwise use `any`"
    },
    {
        "text": "### User:\nExplain the difference between `interface` and `type` in TypeScript\n### Assistant:\nBoth define the shape of an object. The key differences:\n\n**1. Extension syntax:**\n```typescript\n// interface — uses extends\ninterface Animal { name: string; }\ninterface Dog extends Animal { breed: string; }\n\n// type — uses intersection &\ntype Animal = { name: string };\ntype Dog = Animal & { breed: string };\n```\n\n**2. Declaration merging (interface only):**\n```typescript\ninterface Window { myProp: string; } // declare in one file\ninterface Window { otherProp: number; } // merge in another file\n// Window now has both — useful for augmenting library types\n```\n\n**3. Type can do things interface cannot:**\n```typescript\n// Union types\ntype ID = string | number;\n\n// Mapped types\ntype Optional<T> = { [K in keyof T]?: T[K] };\n\n// Conditional types\ntype IsString<T> = T extends string ? true : false;\n```\n\n**Rule of thumb:**\n- Use `interface` for object shapes that might be extended or augmented\n- Use `type` for everything else (unions, mapped types, aliases)"
    },
    # React / Frontend
    {
        "text": "### User:\nWhat is the difference between `useEffect` and `useLayoutEffect` in React?\n### Assistant:\nBoth run side effects, but they fire at different times relative to the DOM.\n\n**`useEffect`** — runs after the browser has painted:\n```javascript\nuseEffect(() => {\n  // Fires after render is committed to screen\n  // Safe for: data fetching, subscriptions, logging\n  fetchData();\n}, [dependency]);\n```\n\n**`useLayoutEffect`** — runs synchronously after DOM mutations, before paint:\n```javascript\nuseLayoutEffect(() => {\n  // Fires before the browser paints\n  // Safe for: reading/writing DOM measurements, preventing flicker\n  const height = ref.current.getBoundingClientRect().height;\n  setHeight(height);\n}, []);\n```\n\n**Timeline:**\n1. React renders → commits DOM changes\n2. `useLayoutEffect` fires (synchronous)\n3. Browser paints\n4. `useEffect` fires (asynchronous)\n\n**When to use which:**\n- `useEffect` — 95% of the time (async, non-blocking, default choice)\n- `useLayoutEffect` — only when you need to measure the DOM or prevent visual flicker"
    },
    {
        "text": "### User:\nExplain React's reconciliation algorithm\n### Assistant:\nReconciliation is how React decides what actually changed between renders and updates the DOM minimally.\n\n**The core idea:**\nInstead of re-rendering the entire DOM on every state change, React compares a virtual DOM snapshot (before) with the new one (after) and only applies the diff.\n\n**Two rules that make it fast:**\n\n1. **Different element types → destroy and rebuild:**\n```jsx\n// React tears down <Counter> and mounts <Profile>\n// (even if they look similar internally)\nbefore: <Counter />\nafter:  <Profile />\n```\n\n2. **Same type → update in place:**\n```jsx\n// React updates className, no unmount/remount\nbefore: <div className=\"old\" />\nafter:  <div className=\"new\" />\n```\n\n**Why `key` matters in lists:**\n```jsx\n// Without key: React matches by position\n// Insert at top → EVERYTHING re-renders\n\n// With key: React tracks identity\n// Insert at top → only new item renders\n<li key={item.id}>{item.name}</li>\n```\n\n**Takeaway:** Keep component types stable, always key lists with stable IDs (not index)."
    },
    # Python
    {
        "text": "### User:\nTeach me how decorators work in Python\n### Assistant:\nA decorator wraps a function to add behavior without modifying its code.\n\n**Step 1 — Functions are objects in Python:**\n```python\ndef greet():\n    return 'hello'\n\nprint(greet)       # <function greet at 0x...>\nprint(greet())     # 'hello'\n```\n\n**Step 2 — A function that wraps another function:**\n```python\ndef make_loud(fn):\n    def wrapper(*args, **kwargs):\n        print('Calling:', fn.__name__)\n        result = fn(*args, **kwargs)\n        print('Done')\n        return result\n    return wrapper\n\ndef greet(name):\n    return f'hello {name}'\n\nloud_greet = make_loud(greet)\nloud_greet('Alice')  # prints: Calling: greet, hello Alice, Done\n```\n\n**Step 3 — The `@` syntax is just shorthand:**\n```python\n@make_loud          # same as: greet = make_loud(greet)\ndef greet(name):\n    return f'hello {name}'\n```\n\n**Real-world example — timing a function:**\n```python\nimport time\nfrom functools import wraps\n\ndef timer(fn):\n    @wraps(fn)  # preserves fn.__name__ and __doc__\n    def wrapper(*args, **kwargs):\n        start = time.perf_counter()\n        result = fn(*args, **kwargs)\n        print(f'{fn.__name__} took {time.perf_counter() - start:.3f}s')\n        return result\n    return wrapper\n\n@timer\ndef slow_computation(n):\n    return sum(range(n))\n```"
    },
    {
        "text": "### User:\nWhat is the difference between `__str__` and `__repr__` in Python?\n### Assistant:\nBoth return string representations of an object, but serve different audiences.\n\n**`__repr__`** — for developers, should be unambiguous:\n```python\nclass Point:\n    def __init__(self, x, y):\n        self.x = x\n        self.y = y\n\n    def __repr__(self):\n        return f'Point({self.x!r}, {self.y!r})'\n\np = Point(1, 2)\nrepr(p)  # 'Point(1, 2)' — ideally, eval(repr(p)) == p\n```\n\n**`__str__`** — for end users, should be readable:\n```python\n    def __str__(self):\n        return f'({self.x}, {self.y})'\n\nstr(p)   # '(1, 2)'\nprint(p) # (1, 2)\n```\n\n**Fallback rule:** If `__str__` is missing, Python falls back to `__repr__`. So always define `__repr__` at minimum.\n\n**In practice:**\n```python\n>>> p         # uses __repr__ (in REPL)\nPoint(1, 2)\n>>> print(p)  # uses __str__\n(1, 2)\n>>> [p]       # list uses __repr__ for items\n[Point(1, 2)]\n```"
    },
    # Data structures & algorithms
    {
        "text": "### User:\nExplain binary search step by step and write it in JavaScript\n### Assistant:\nBinary search finds a target in a **sorted** array in O(log n) time — much faster than scanning every element.\n\n**Core idea:** At each step, eliminate half the remaining elements.\n\n**Step by step on `[1, 3, 5, 7, 9]`, target = 7:**\n```\nStep 1: left=0, right=4, mid=2 → arr[2]=5 < 7 → search right half\nStep 2: left=3, right=4, mid=3 → arr[3]=7 = 7 → found!\n```\n\n**Implementation:**\n```javascript\nconst binarySearch = (arr, target) => {\n  let left = 0;\n  let right = arr.length - 1;\n\n  while (left <= right) {\n    const mid = Math.floor((left + right) / 2);\n\n    if (arr[mid] === target) return mid;       // found\n    if (arr[mid] < target)  left = mid + 1;   // target is right\n    else                    right = mid - 1;  // target is left\n  }\n\n  return -1; // not found\n};\n\nbinarySearch([1, 3, 5, 7, 9], 7); // 3 (index)\nbinarySearch([1, 3, 5, 7, 9], 4); // -1\n```\n\n**Why O(log n)?** Each iteration halves the search space. 1000 elements → max 10 iterations."
    },
    {
        "text": "### User:\nWhat is Big O notation? Explain with simple examples.\n### Assistant:\nBig O describes how an algorithm's runtime grows as input size grows. It answers: \"if I double the input, what happens to the time?\"\n\n**O(1) — Constant:** Input size doesn't matter\n```javascript\nconst getFirst = (arr) => arr[0]; // always one step\n```\n\n**O(n) — Linear:** Time grows proportionally\n```javascript\nconst findMax = (arr) => {\n  let max = arr[0];\n  for (const x of arr) if (x > max) max = x; // touches every element once\n  return max;\n};\n```\n\n**O(n²) — Quadratic:** Nested loops over the same data\n```javascript\nconst hasDuplicate = (arr) => {\n  for (let i = 0; i < arr.length; i++)\n    for (let j = i + 1; j < arr.length; j++)\n      if (arr[i] === arr[j]) return true;\n  return false;\n};\n```\n\n**O(log n) — Logarithmic:** Halves the problem each step (binary search)\n\n**O(n log n) — Common in sorting** (merge sort, quicksort average case)\n\n**Quick comparison for n=1000:**\n| Notation | Operations |\n|----------|------------|\n| O(1) | 1 |\n| O(log n) | ~10 |\n| O(n) | 1,000 |\n| O(n log n) | ~10,000 |\n| O(n²) | 1,000,000 |"
    },
    # System design / architecture
    {
        "text": "### User:\nWhat is the difference between SQL and NoSQL databases? When would you use each?\n### Assistant:\n**SQL (relational):** Data in tables with fixed schemas, rows, and columns. Relationships enforced by foreign keys.\n\n```sql\n-- Strongly typed, structured\nCREATE TABLE users (\n  id UUID PRIMARY KEY,\n  email TEXT UNIQUE NOT NULL,\n  created_at TIMESTAMP DEFAULT NOW()\n);\n```\n\n**NoSQL:** Flexible schema, various data models (document, key-value, graph, column-family).\n\n```json\n// MongoDB document — fields can vary per document\n{ \"_id\": \"abc\", \"email\": \"a@b.com\", \"preferences\": { \"theme\": \"dark\" } }\n```\n\n**Use SQL when:**\n- Data has clear relationships (users → orders → products)\n- You need ACID transactions (financial, inventory)\n- Schema is stable\n- Examples: PostgreSQL, MySQL, SQLite\n\n**Use NoSQL when:**\n- Schema changes frequently or varies per record\n- Massive horizontal scale needed (millions of writes/sec)\n- Data is hierarchical/nested naturally\n- Examples: MongoDB (documents), Redis (cache/key-value), DynamoDB (scale)\n\n**Common mistake:** Don't pick NoSQL just because it feels \"modern.\" PostgreSQL with JSONB columns handles most flexible-schema needs while keeping ACID guarantees."
    },
    {
        "text": "### User:\nExplain REST vs GraphQL — when would you choose one over the other?\n### Assistant:\n**REST:** Multiple endpoints, each returns a fixed shape.\n```\nGET /users/123          → full user object\nGET /users/123/posts    → all posts (even if you only need titles)\nPOST /users             → create user\n```\n\n**GraphQL:** Single endpoint, client specifies exactly what it needs.\n```graphql\nquery {\n  user(id: \"123\") {\n    name              # only what I need\n    posts { title }   # no over-fetching\n  }\n}\n```\n\n**Choose REST when:**\n- Simple CRUD operations\n- Public API (REST is universally understood)\n- Caching matters (REST URLs cache naturally at CDN/browser level)\n- Small team, fast iteration\n\n**Choose GraphQL when:**\n- Multiple clients with different data needs (mobile vs web)\n- You're suffering from over-fetching (too much data) or under-fetching (too many round trips)\n- Rapid frontend iteration without backend changes\n- Complex interconnected data (social graphs, content platforms)\n\n**Reality check:** Most apps do fine with REST. Only switch to GraphQL when you're actually feeling the pain it solves."
    },
]

def build_dataset(output_dir: str, train_ratio=0.6, valid_ratio=0.27):
    random.seed(42)
    random.shuffle(EXAMPLES)

    n = len(EXAMPLES)
    train_n = int(n * train_ratio)
    valid_n = int(n * valid_ratio)

    splits = {
        "train": EXAMPLES[:train_n],
        "valid": EXAMPLES[train_n:train_n + valid_n],
        "test": EXAMPLES[train_n + valid_n:],
    }

    out = Path(output_dir)
    out.mkdir(parents=True, exist_ok=True)

    for name, examples in splits.items():
        path = out / f"{name}.jsonl"
        with open(path, "w") as f:
            for ex in examples:
                f.write(json.dumps(ex) + "\n")
        print(f"{name}: {len(examples)} examples → {path}")

    print(f"\nTotal: {n} examples across {output_dir}")


if __name__ == "__main__":
    build_dataset("./clean_data")
