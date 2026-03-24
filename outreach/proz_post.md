# ProZ Forum Post

**Forum:** Machine translation and post-editing (or Technology)
**Subject:** New DE→EN patent translation tool with automatic quality triage — looking for feedback

---

Hi everyone,

I'm a linguist (MA, Heidelberg) who's been working on a specialized tool for German-to-English patent translation. I wanted to share it here because I think it addresses a real pain point in patent PE workflows, and I'd genuinely appreciate feedback from translators and PMs who work in this space.

**What it does:**

GlossWerk is a translation pipeline that combines AI translation with automatic quality estimation. Instead of reviewing every segment equally, it triages each one:

- Green: publishable as-is (in our testing, this covers about two-thirds of segments)
- Orange: minor issue flagged, one-click fix available
- Red: needs real attention, with a detailed explanation of what's wrong

It also handles terminology extraction up front — you review and confirm key terms before translation starts, and those terms are enforced throughout the document. For anyone who's dealt with German patent syntax (nested participials, compound chains, nominalizations that span half a page), it includes structural analysis that flags and restructures those constructions.

**Why I built it:**

I spent enough time doing DE→EN patent PE to know that most of the time is spent on segments that don't need it. The quality estimation layer is specifically trained on patent-typical errors: calques from German, lexical redundancy from morphologically related terms, and readability issues from literal clause structure transfer.

**What I'm looking for:**

I'm offering free demo access (3 patents) to translators and LSPs who work with German patents. I'm not trying to sell anything right now — I want real feedback on whether this actually saves time in practice and where it falls short.

If you're interested, visit glosswerk.com or send me a message and I'll set you up with an invite code.

Matt Gorman
mjg@glosswerk.com
