# Coding Game Winter 2026

This repo contains the code for competing in [Coding Game Winter 2026](https://www.codingame.com/ide/challenge/winter-challenge-2026-exotec)

# Winter Challenge 2026: SNAKEBYTE

**CodinGame Winter Challenge 2026**  
*10D 11H 19MN 08SC remaining*  
**Bronze League**

## 🎯 Goal
Snag power sources to grow your snake-like robots and ensure you have the biggest bots standing.

## 🗺️ Map & Mechanics
The game unfolds on a **side-view grid** with **impassable platforms (#)** and **free cells (.)**.

**Key Elements:**
- **Snakebots**: Multiple adjacent cells, **head first** in the chain
- **Gravity**: Snakebots fall until supported by platforms, power sources, or other snakebots
- **Perpetual Motion**: Move in last direction (starts **UP**) unless redirected

## 🐍 Snakebot Movement
**Every turn**:
1. Head advances in current direction
2. Body follows head's previous position
3. **Collisions resolved simultaneously**:

| Case | Target Cell Contains | Effect |
|------|-------------------|--------|
| **Case 1** | Platform or body part | **Head destroyed** (next part becomes head if ≥3 parts remain, else full removal) |
| **Case 2** | **Power source** | **EATEN**: Grow (new tail), cell becomes passable |
| **Special** | Multiple heads on power source | **All eat** simultaneously! |

4. **Gravity**: All fall downward until supported
5. **Border Fall**: Snakebots extending beyond grid are **removed**

## 🎮 Actions
Output **one line** with actions separated by `;`:

**Movement Commands** (for snakebot `id`):
```
id UP     // (0,-1)
id DOWN   // (0,1) 
id LEFT   // (-1,0)
id RIGHT  // (1,0)
```
*Optional debug text*: `1 LEFT [debug note]`

**Debug Commands**:
```
MARK x y  // Up to 4 per turn
WAIT      // Do nothing
```

**Example**: `1 LEFT; 2 RIGHT; MARK 12 2`

## ⛔ Game End Conditions
Game ends when **any** are true:
- All your snakebots destroyed
- No power sources remain
- **200 turns** passed

## 🏆 Victory Conditions
**Most total body parts** across all your snakebots.

## 💥 Defeat Conditions
- No command within time limit
- Invalid command syntax

## 📋 Game Protocol

### Initialization (First Turn)
```
Line 1: myId (0 or 1)
Line 2: width (15-45)
Line 3: height (10-30)
Next height lines: Grid (#=platform, .=empty)
Line: snakbotsPerPlayer
Next snakbotsPerPlayer lines: YOUR snakebotIds
Next snakbotsPerPlayer lines: OPPONENT snakebotIds
```

### Per-Turn Input
```
Line 1: powerSourceCount
Next powerSourceCount lines: x y
Line: snakebotCount (1-8)
Next snakebotCount lines: snakebotId "x1,y1:x2,y2:..."
```

**Body Example**: `"0,1:1,1:2,1"` = head@(0,1), body@(1,1), tail@(2,1)

## ⚙️ Technical Constraints
| Limit | Value |
|-------|-------|
| **Turn response** | ≤50ms |
| **First turn** | ≤1000ms |
| **Width** | 15-45 |
| **Height** | 10-30 |
| **Snakebots** | 1-8 total |

**Source**: [GitHub repo](https://github.com/CodinGame/WinterChallenge2026-Exotec)

## 🐛 Debugging Tips
- `MARK x y` (max 4/turn)
- Hover grid for cell info
- Gear icon: viewer options
- Keyboard: space (play/pause), arrows (step)

***

**Perfect for your Rust CLAUDE.md template** – gravity, collisions, heuristics galore! Ready to build the bot?
