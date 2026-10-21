# COMP7503 Multimedia Technologies: Midterm Teaching Notes
# COMP7503 多媒體技術：期中試溫習教學筆記

**Coverage / 涵蓋範圍:** Lecture PDFs 01-05, academic year 2026-2027; 231 PDF pages in total. / 2026-2027 學年，第 01 至 05 份講義，共 231 個 PDF 頁面。

**Language / 語言:** English explanations followed by equivalent Traditional Chinese explanations. / 每項英文解說後附相應繁體中文解說。

**Important / 重要說明:** These are teaching and revision notes based on the supplied lectures, not an official examination specification or a prediction of questions. Definitions, pipelines, and examples from the slides are expanded into explanations. Material labelled **Supplement** adds background or derivations; **Clarification** identifies a simplification, historical claim, or ambiguity in the slides. Such additions are not evidence of extra examinable syllabus content.

本筆記根據提供的講義編寫，用於教學及溫習，並非官方考試範圍或試題預測。筆記把投影片的定義、流程及例子展開說明。標示 **補充** 的內容屬背景知識或推導；標示 **釐清** 的內容用來解釋講義中的簡化、歷史性說法或歧義。這些延伸不代表老師已指定額外考核內容。

## Contents / 目錄

1. [How the course fits together / 全課的整體脈絡](#overview)
2. [Lecture 01: Introduction / 導論](#lecture-01)
3. [Lecture 02: Elements of Multimedia Computing / 多媒體運算的要素](#lecture-02)
4. [Lecture 03: Example Applications / 應用實例](#lecture-03)
5. [Lecture 04: Sensors, Digitization, and Sound / 感測器、數碼化與聲音](#lecture-04)
6. [Lecture 05: Reproduction, Speech, Music, Light, and Vision / 重現、語音、音樂、光與視覺](#lecture-05)
7. [Formula sheet and worked examples / 公式表及計算例題](#formulas)
8. [Practice questions with model answers / 練習題及參考答案](#practice)
9. [Final revision checklist / 最後溫習清單](#quick-review)
10. [Bilingual terminology / 雙語詞彙表](#glossary)
11. [Source coverage map / 來源涵蓋表](#coverage)

<a id="overview"></a>
## 0. How the Course Fits Together / 全課的整體脈絡

### 0.1 Source Lectures / 原始講義

Page references below mean **PDF page positions, counting the first page as 1**, not necessarily the printed slide numbers. In particular, lecture 02 has gaps in its printed numbering.

下文頁碼一律指 **PDF 的實際頁次，第一頁算第 1 頁**，不一定等於投影片頁腳印出的數字；第 02 份講義的頁腳編號尤其有跳號。

| Code / 代號 | Source / 來源 | PDF Pages / 頁數 |
|---|---|---:|
| L01 | [01 - Introduction.pdf](01%20-%20Introduction.pdf) | 35 |
| L02 | [02 - Elements of Multimedia Computing.pdf](02%20-%20Elements%20of%20Multimedia%20Computing.pdf) | 22 |
| L03 | [03 - Example Applications.pdf](03%20-%20Example%20Applications.pdf) | 63 |
| L04 | [04 - Media Sensor and Reproduction - 1.pdf](04%20-%20Media%20Sensor%20and%20Reproduction%20-%201.pdf) | 49 |
| L05 | [05 - Media Sensor and Reproduction - 2.pdf](05%20-%20Media%20Sensor%20and%20Reproduction%20-%202.pdf) | 62 |

### 0.2 The Main Story / 主線

**EN:** A physical event produces signals. Sensors capture incomplete observations; digitization makes them computable. Processing extracts features, while context and prior knowledge help connect features to objects and events. Combining correlated observations can produce useful insights. Displays, speakers, and other actuators then present the result so the user can understand, explore, and respond.

**繁中：** 真實事件產生訊號。感測器擷取不完整的觀測資料，再經數碼化成為電腦可以處理的形式。系統抽取特徵，並利用情境及先驗知識把特徵連結到物件和事件。整合有關聯的觀測後，便可能產生有用的洞見。最後，顯示器、揚聲器及其他致動器把結果呈現給使用者，讓人理解、探索及回應。

```text
Physical world / 真實世界
  -> sensors / 感測器
  -> sampling + quantization / 取樣 + 量化
  -> digital streams + metadata / 數碼資料流 + 中繼資料
  -> features + context + prior knowledge / 特徵 + 情境 + 先驗知識
  -> integration and interpretation / 整合及詮釋
  -> information and insights / 資訊及洞見
  -> presentation and user feedback / 呈現及使用者回饋
  -> further exploration / 進一步探索
```

**Running example / 貫穿例子:** In an elderly-care system, a camera observes a person falling, an accelerometer detects sudden movement, a microphone detects a cry, and a heart-rate sensor provides physiological context. Combining them can support an alert more reliably than one stream alone, although uncertainty and false alarms remain. / 長者照護系統可透過攝影機觀察跌倒、加速度計偵測急速動作、麥克風接收呼叫，以及心率感測器提供生理情境。整合後可比單一資料流更可靠地支援警報，但仍有不確定性和誤報。

### 0.3 How to Revise / 溫習方法

**EN:** For each topic, practise four things: define it; draw or explain its pipeline; compare it with a nearby concept; apply it to a concrete scenario. For calculations, state assumptions and units before substituting values. For application questions, explain what each input contributes and what information is lost.

**繁中：** 每個主題都練習四件事：下定義、畫出或解釋流程、與相近概念比較、套用到具體情境。計算題先寫假設和單位，再代入數值。應用題須交代每種輸入提供甚麼，以及哪些資訊會流失。

<a id="lecture-01"></a>
## 1. Lecture 01: Introduction / 導論

### 1.1 What Is Multimedia? / 甚麼是多媒體？

**Source / 來源：L01 pp. 2-8.**

**EN:** Multimedia is the combined use of several media to stimulate one or more senses and deliver relevant information. Human senses are sight, hearing, touch, smell, and taste. Each supplies only partial information. A sentence alone may be ambiguous; an image, tone of voice, or setting can disambiguate it. The lecture's dentist example shows that the same words can be interpreted very differently without context.

**繁中：** 多媒體是結合多種媒介，刺激一種或多種感官，以傳遞相關資訊。人類五感包括視覺、聽覺、觸覺、嗅覺和味覺；每種感官只能提供部分資訊。單獨一句話可能有歧義，但影像、語氣或場景能幫助消除歧義。講義的牙醫例子說明：沒有情境，同一句說話可以有完全不同的解讀。

**EN:** Audio and vision dominate practical multimedia because humans analyse them effectively and technology can capture, process, and reproduce them relatively well. Smell and taste are harder to communicate electronically. Multimedia is therefore broader than entertainment, music, video playback, or a web page.

**繁中：** 聲音和視覺在實際多媒體中佔主導地位，因為人類善於分析這兩類訊息，而科技亦較能有效地擷取、處理和重現它們。嗅覺和味覺則較難透過電子系統傳遞。因此，多媒體不只等於娛樂、音樂、播放影片或網頁。

### 1.2 Multimedia Computing / 多媒體運算

**EN:** Computing is goal-oriented activity involving computing machinery. Multimedia computing uses signals from diverse sensors and presents relevant information through suitable sensory modalities. The two broad goals in the lecture are **sharing information about experiences** and **extracting knowledge from experiences**.

**繁中：** 運算是涉及電腦設備、並以達成目標為本的活動。多媒體運算利用不同感測器的訊號，再以適當感官媒態呈現相關資訊。講義指出兩個主要目標：**分享與經驗有關的資訊**，以及 **從經驗抽取知識**。

**Exam distinction / 答題區分:** A page that places unrelated text, music, and pictures together uses several media, but it does not necessarily perform meaningful multimedia integration. A traffic system that combines video, GPS, and time to infer congestion demonstrates the lecture's stronger computing perspective. / 把互不相關的文字、音樂及圖片放在一起，雖然使用多種媒介，卻不一定做到有意義的資訊整合。把影片、GPS 和時間結合以推斷交通擠塞，才更能體現講義強調的多媒體運算。

### 1.3 Communication Across Time, Space, and Addressees / 跨時間、空間及受眾的通訊

**Source / 來源：L01 pp. 9-16.**

**EN:** Communication shares experiences and preserves, creates, and distributes knowledge. Technologies remove different constraints. **Time invariance** means information can be accessed at a different time; **space invariance** means it can cross locations; **addressee invariance** concerns reaching more or different recipients. These are conceptual capabilities, not promises of unlimited delivery.

**繁中：** 通訊讓人分享經驗，並保存、創造和傳播知識。不同技術解除不同限制。**時間不變性／跨時間能力** 指資訊可在其他時間被接收；**空間不變性／跨空間能力** 指資訊可跨地點傳遞；**受眾不變性／擴展受眾能力** 關乎接觸更多或不同接收者。這些是概念上的能力，並不代表傳遞完全不受限制。

| Medium or invention / 媒介或發明 | Main contribution / 主要貢獻 | Constraint reduced / 減少的限制 |
|---|---|---|
| Spoken language / 口語 | Immediate natural expression / 即時、自然表達 | None intrinsically without recording or transmission / 本身沒有錄製或傳輸時仍受各項限制 |
| Writing / 文字 | Persistent symbolic record / 持久的符號記錄 | Time / 時間 |
| Paper / 紙張 | Portable records / 可攜帶的記錄 | Time and space / 時間及空間 |
| Printing / 印刷 | Cheap replication and mass distribution / 大量複製及分發 | Time, space, addressees / 時間、空間及受眾 |
| Telegraph / 電報 | Remote symbolic messages / 遠距離符號訊息 | Space / 空間 |
| Telephone / 電話 | Remote natural speech / 遠距離自然語音 | Space / 空間 |
| Radio / 廣播 | Wireless one-to-many audio / 無線一對多聲音傳播 | Space and addressees / 空間及受眾 |
| Television / 電視 | Broadcasting moving images and sound / 廣播動態影像及聲音 | Space and addressees / 空間及受眾 |
| Movies and recording media / 電影及錄製媒體 | Replayable audiovisual experience / 可重播的視聽經驗 | Time, distribution across space and audiences / 時間，並可跨地點及受眾分發 |
| Digital media / 數碼媒體 | Machine processing, copying, storage / 電腦處理、複製及儲存 | Time, space, addressees / 時間、空間及受眾 |
| Internet / 互聯網 | Networked access and personalized reception / 網絡存取及個人化接收 | Time, space, and flexible recipients / 時間、空間及彈性受眾 |

**Example / 例子:** A live phone call crosses space but not time unless recorded. A recorded lecture crosses time; uploading it also makes distribution across space and to many learners easier. / 即時電話可跨空間，但沒有錄音便不能跨時間。錄製課堂可跨時間；上載到網絡後，更方便跨地點向大量學生分發。

### 1.4 Why Multimedia Became Practical / 多媒體為何逐漸變得可行？

**Source / 來源：L01 pp. 17-24.**

**EN:** Early computers were limited by processing power, storage capacity, memory, graphics, bandwidth, and algorithms. Before the 1990s, text and numeric workloads were much easier than real-time audio/video. Early-1990s PCs added better processors, colour graphics, and sound cards. Mid-1990s machines combined graphical operating systems, larger disks, CD-ROMs, and integrated stereo sound. By the late 1990s, multimedia was increasingly a normal computer capability rather than a special label.

**繁中：** 早期電腦受運算能力、儲存容量、記憶體、圖像能力、頻寬和演算法限制。1990 年代以前，處理文字和數值遠比即時聲音或影片容易。1990 年代初的個人電腦逐步加入較快處理器、彩色圖像和音效卡；年代中期再結合圖像化作業系統、大容量硬碟、光碟機及整合式立體聲。到年代末，多媒體逐漸成為電腦的基本能力，而不是特別標籤。

**EN:** Mobile devices extend this shift: they combine cameras, microphones, touch input, wireless networking, positioning, and motion sensors. Applications also drive hardware: numerical work motivates arithmetic units; business applications motivate databases and servers; web applications motivate connectivity; social and mobile applications motivate media processing and location sensing.

**繁中：** 流動裝置進一步推動轉變，結合攝影機、麥克風、觸控輸入、無線網絡、定位及動作感測器。應用亦反過來推動硬件：數值運算促進算術單元；商業應用促進資料庫和伺服器；網頁應用促進連線；社交及流動應用促進媒體處理和定位技術。

**Revision focus / 溫習重點:** Explain the cause-and-effect relationship between resource improvements and new applications. Exact historical product specifications are examples, not universal boundaries. / 重點是解釋資源改善如何帶來新應用。歷史產品規格只是例子，並非普遍技術界限。

### 1.5 Emerging Applications and Evolving Information / 新興應用與不斷演變的資訊

**Source / 來源：L01 pp. 25-27.**

**EN:** Modern applications increasingly use live, spatiotemporal streams rather than only static records. They need a holistic event view instead of isolated data silos. Users want relevant insights regardless of which medium or sensor produced them, and often explore interactively rather than submit one fixed database query.

**繁中：** 現代應用愈來愈依賴即時、具時空性的資料流，而不只是靜態記錄。系統需要事件的整體圖像，而非彼此隔離的資料孤島。使用者關心有用洞見，不一定在意資料來自哪種媒介或感測器；互動方式亦往往是探索，而非一次過提交固定資料庫查詢。

**EN:** Compare the lecture's e-commerce questions: counting orders between Hong Kong and Taiwan is a well-defined aggregation once direction and period are fixed; identifying rapidly growing regions requires comparison over time and a definition of growth; finding the hottest item requires a time window and a definition of popularity. The latter questions require more context and interpretation.

**繁中：** 比較講義的電子商貿問題：計算香港與台灣之間的訂單量，只要方向及時段已指定，便是明確的彙總；找出增長迅速的地區，需要跨時間比較和「增長」的定義；找出最熱門商品，則需要時窗和「熱門」的準則。後兩者更依賴情境和詮釋。

### 1.6 Experiential Environments / 體驗式環境

**Source / 來源：L01 pp. 28-32.**

**EN:** An experiential environment lets users directly use their senses to inspect information about an event and interact according to their interests. It aims at **human-machine synergy**: humans contribute perceptual and conceptual reasoning; computers contribute fast mathematical and logical processing.

**繁中：** 體驗式環境讓使用者直接透過感官觀察事件資訊，並按自身興趣互動。目的是達成 **人機協同**：人類提供知覺及概念推理能力，電腦提供快速數學與邏輯處理能力。

| Property / 特性 | Meaning / 意義 | Example / 例子 |
|---|---|---|
| Direct and intuitive / 直接及直觀 | Interact without unfamiliar command systems / 不必依賴陌生指令系統 | Swipe through photographs / 滑動瀏覽照片 |
| Same query and presentation spaces / 查詢與呈現使用同一空間 | Modify the displayed representation and observe its consequences / 直接修改顯示內容並觀察後果 | Edit a spreadsheet cell and see related results / 修改試算表儲存格並觀看相關結果 |
| User state and context / 使用者狀態及情境 | Adapt interpretation and interaction to what the user is doing / 配合使用者當下活動來呈現及互動 | A game combining visual, audio, and tactile feedback / 結合視覺、聲音及觸覺回饋的遊戲 |

**Exam trap / 常見陷阱:** An experiential environment is not defined by having impressive graphics. It is defined by direct perception, contextual interaction, and effective exploration. / 體驗式環境不是由華麗圖像來定義，而是由直接感知、情境化互動和有效探索來定義。

### 1.7 Formal Model of Multimedia Computing / 多媒體運算的形式化模型

**Source / 來源：L01 pp. 33-35.**

**EN:** Let $S_1,\ldots,S_n$ be synchronized sensor streams. The $n$ streams may belong to $K$ data types; $n$ and $K$ are not necessarily equal. Each stream $S_i$ has metadata $M_i$ and feature streams $F_{ij}$, where $j$ indexes features extracted from that stream. Integration uses features and metadata to generate insights for a user. User feedback becomes another input stream.

**繁中：** 設 $S_1,\ldots,S_n$ 為已同步的感測資料流。這 $n$ 條資料流可屬於 $K$ 種資料類型；$n$ 不一定等於 $K$。每條 $S_i$ 帶有中繼資料 $M_i$，並可抽取特徵流 $F_{ij}$，其中 $j$ 表示該資料流的第幾種特徵。系統整合特徵和中繼資料，產生供使用者使用的洞見。使用者的回饋亦可成為另一條輸入流。

**Example / 例子:** For a trip, photos are visual content; timestamps and GPS coordinates are metadata; detected landmarks are features; the reconstructed itinerary is an insight. Two cameras give two streams but may still have only one media type. / 旅行中，照片屬視覺內容；時間戳記及 GPS 座標屬中繼資料；辨認出的地標屬特徵；重建的行程路線屬洞見。兩部攝影機提供兩條資料流，但可能只涉及一種媒體類型。

**EN:** Synchronization matters because streams must refer to the same event interval. A cry recorded now and a fall captured yesterday should not be fused as evidence of one incident. Metadata also needs consistency: time zones, clock offsets, units, and locations can affect the result.

**繁中：** 同步非常重要，因為資料流必須對應同一事件時段。現在錄得的呼叫不能與昨天拍到的跌倒直接融合成同一事故證據。中繼資料亦須一致：時區、時鐘偏移、單位及位置都會影響結果。

**One-sentence answer / 一句總結:** Multimedia computing combines partial, correlated observations from multiple sources in the context of a problem to produce relevant information and insights. / 多媒體運算在問題情境下，結合來自多個來源、彼此相關但不完整的觀測，以產生相關資訊及洞見。

<a id="lecture-02"></a>
## 2. Lecture 02: Elements of Multimedia Computing / 多媒體運算的要素

### 2.1 Objective and Central Challenge / 目標與核心挑戰

**Source / 來源：L02 p. 2.**

**EN:** A multimedia system should process different data streams **as one correlated set** representing knowledge relevant to a problem. The challenge is to discover useful correlations and combine partial evidence into a holistic account. Simply collecting more data does not guarantee understanding: the relationships must be meaningful in the application context.

**繁中：** 多媒體系統應把不同資料流視為 **一組互相關聯的資料** 來處理，代表與問題有關的資訊和知識。難點是找出有用關聯，並把部分證據組成整體理解。單純收集更多資料不保證理解得更好；資料之間的關係必須在應用情境中有意義。

### 2.2 Experience, Information, Knowledge, and Communication / 經驗、資訊、知識與通訊

**Source / 來源：L02 pp. 3-6.**

| Concept / 概念 | Explanation / 解釋 |
|---|---|
| Experience / 經驗 | Direct observation of, or participation in, events / 親自觀察或參與事件 |
| Information / 資訊 | An efficient but abstract communication of experience / 有效率但抽象化的經驗表達 |
| Knowledge / 知識 | Understanding accumulated and organized from experiences / 從經驗累積及組織而成的理解 |
| Communication / 通訊 | Sharing information about experiences across people, time, or space / 跨人物、時間或空間分享與經驗有關的資訊 |

**EN:** A description does not transfer the full experience of being present. It selects and abstracts. Relevant shared experiences help recipients interpret the information. Multimedia can restore some sensory context, but it still cannot guarantee an identical subjective experience.

**繁中：** 描述不能把「親身在場」的完整經驗直接移交給別人；它必然經過選擇和抽象化。相關的共同經驗有助接收者理解資訊。多媒體可以補回部分感官情境，但仍不保證每個人產生相同的主觀體驗。

**Teaching example / 教學例子:** Reading that an alarm was loud differs from hearing a recording; hearing it differs again from experiencing it during an emergency. / 讀到「警報很大聲」、聽到警報錄音，以及身處緊急情況下聽到警報，是三種不同層次。

### 2.3 Dictionaries and Percepts / 字典與知覺單元

**Source / 來源：L02 p. 7.**

**EN:** Communication requires a shared dictionary: agreed symbols and meanings. Natural language uses words; computer systems may use codes, terms, and keys; some compression methods also use dictionaries of reusable patterns. In multimedia, a basic perceived element is a **percept**. A visual dictionary may contain lines, curves, shapes, or learned visual patterns.

**繁中：** 通訊需要共同字典，即彼此同意的符號及其意義。自然語言使用詞語；電腦系統可使用代碼、術語和鍵值；部分壓縮方法亦使用可重用圖樣的字典。多媒體中的基本感知元素稱為 **知覺單元（percept）**。視覺字典可包含線、曲線、形狀，或學習得到的視覺圖樣。

**Distinction / 區分:** A percept is not automatically a semantic object. A curved edge is a low-level pattern; identifying it as part of a cup requires interpretation. / 知覺單元不會自動等同於有語意的物件。「彎曲邊緣」是低階圖樣；認出它屬於杯子的一部分，還需要詮釋。

### 2.4 Objects and Events / 物件與事件

**Source / 來源：L02 p. 8.**

**EN:** Objects model relatively static entities; events model changes in relationships among objects over time. Audio and video are inherently dynamic: they record attributes as functions of time. For example, a person and a door are objects; the person opening the door is an event. A single frame may show objects but not establish the event's temporal order.

**繁中：** 物件用來表示相對靜態的實體；事件則表示物件之間的關係隨時間改變。聲音和影片本質上具有動態性，記錄的是隨時間變化的屬性。例如，人和門是物件；「人把門打開」是事件。單一畫面可能顯示物件，卻未必足以確立事件的先後次序。

### 2.5 Perception Requires Prior Knowledge / 知覺需要先驗知識

**Source / 來源：L02 pp. 9-13, 15-16.**

**EN:** Perception interprets sensory signals to recover useful information. Its output depends on both the signal and existing knowledge. The lecture describes it as generating hypotheses and selecting interpretations best supported by evidence. The hidden Dalmatian image illustrates that recognition becomes easier once the observer knows what to look for; the wine/soy-sauce example illustrates contextual ambiguity.

**繁中：** 知覺是詮釋感官訊號、恢復有用資訊的過程。輸出同時取決於訊號和既有知識。講義把它描述為產生多個假設，再挑選最受證據支持的解讀。隱藏斑點狗的圖片說明：知道要找甚麼之後，辨認會變容易；紅酒／豉油的例子則說明情境可造成或消除歧義。

**Supplement / 補充:** A useful conceptual expression is $P(H\mid D,C)\propto P(D\mid H,C)P(H\mid C)$: interpretation $H$ depends on observed data $D$ and context $C$. This is a teaching aid, not an equation required by the slides. / 可用 $P(H\mid D,C)\propto P(D\mid H,C)P(H\mid C)$ 幫助理解：解讀 $H$ 取決於觀測資料 $D$ 及情境 $C$。這只是教學輔助，並非投影片指定的公式。

**Exam trap / 常見陷阱:** Prior knowledge guides interpretation; it does not justify ignoring contradictory data. Ambiguous evidence should lead to uncertainty, not an unsupported confident label. / 先驗知識可以引導詮釋，但不能成為忽略矛盾資料的理由。證據含糊時，應承認不確定性，而不是武斷地下結論。

### 2.6 Neisser's Perceptual Cycle / Neisser 知覺循環

**Source / 來源：L02 p. 14.**

**EN:** A schema is an internal model or expectation about the environment. It directs exploration; exploration samples the environment; the available information modifies the schema. Perception is therefore an active loop, not merely a one-way sensor-to-label pipeline.

**繁中：** 基模（schema）是對環境的內在模型或預期。基模引導探索；探索從環境取得資訊；取得的資訊再修改基模。因此，知覺是一個主動循環，而不是單向「感測器輸入、標籤輸出」的流程。

```text
Schema / 基模
  -- directs / 引導 --> Exploration / 探索
  -- samples / 取樣 --> Environment / 環境
  -- modifies / 修改 --> Schema / 基模
```

**Example / 例子:** You suspect a bottle contains soy sauce. You inspect its label or smell it, then revise your interpretation. A robot can similarly move its camera to obtain a less ambiguous view. / 你懷疑瓶內是豉油，於是查看標籤或聞氣味，再更新判斷。機械人亦可移動攝影機，取得較少歧義的角度。

### 2.7 Four System Problems / 四類系統問題

**Source / 來源：L02 p. 17.**

1. **Representation / 表示:** How can data be stored or transmitted compactly? / 如何緊湊地儲存或傳輸資料？
2. **Presentation / 呈現:** How can large amounts of data communicate the intended information clearly? / 如何把大量資料清楚地轉化為原本想傳達的資訊？
3. **System coordination / 系統協調:** How should different data types, rates, formats, and timing requirements be handled? / 如何處理不同資料類型、速率、格式和時間要求？
4. **Integration / 整合:** How can the streams jointly provide information needed to solve the problem? / 如何結合資料流，取得解決問題所需的資訊？

**EN:** These problems are related but distinct. Compression can reduce storage without solving meaning; a clear visualization can improve presentation without fixing incorrect synchronization.

**繁中：** 這些問題彼此相關，卻並不相同。壓縮可以減少儲存量，但不會自動解決語意問題；清晰的視覺化可以改善呈現，卻不能修正錯誤同步。

### 2.8 The Semantic Gap / 語意鴻溝

**Source / 來源：L02 pp. 18-19.**

**EN:** The semantic gap is the difference between what a machine can extract from perceptual data and how a user interprets that same data in a situation. Machines operate on samples, pixels, bits, and features; users ask about objects, events, intentions, and meaning.

**繁中：** 語意鴻溝是機器從感知資料中能抽取的內容，與使用者在特定情境下對同一資料的解讀之間的差距。機器處理樣本、像素、位元和特徵；使用者關心物件、事件、意圖和意義。

**Example / 例子:** A machine detects a red region and movement; a user wants to find a red vehicle that entered a restricted area. Recognition, temporal tracking, zone metadata, and event rules help bridge the gap. / 機器偵測到紅色區域和移動；使用者想找的卻是「駛入禁區的紅色車輛」。物件辨認、時間追蹤、區域中繼資料及事件規則，有助縮窄兩者差距。

**EN:** Increasing resolution alone does not eliminate the gap. It provides more detailed data, not automatically the correct semantic interpretation.

**繁中：** 單純提高解像度不能消除語意鴻溝，因為它只提供更細緻資料，不會自動產生正確語意解讀。

### 2.9 Content, Context, and Metadata / 內容、情境與中繼資料

**Source / 來源：L02 pp. 20-21.**

| Term / 術語 | Meaning / 意義 | Photo example / 照片例子 |
|---|---|---|
| Content / 內容 | The acquired data itself / 擷取的資料本身 | Pixel values depicting a street / 描繪街道的像素值 |
| Context / 情境 | Conditions related to acquisition and interpretation / 與擷取及詮釋有關的條件 | Where, when, why, and under what lighting / 地點、時間、目的及光線 |
| Metadata / 中繼資料 | Structured data describing other data / 描述其他資料的結構化資料 | EXIF timestamp, exposure, camera model, optional GPS / EXIF 時間、曝光、相機型號，以及可能附帶的 GPS |

**EN:** Metadata can provide context, but context is broader than stored metadata. Metadata may be missing, inaccurate, or edited. For example, an EXIF clock can be wrong, and an image may have no location field. Content and context should be considered together.

**繁中：** 中繼資料可以提供情境，但情境不只限於已儲存的中繼資料。中繼資料可能缺失、不準確或被修改，例如 EXIF 時鐘設定錯誤，或影像根本沒有定位欄位。因此，理解多媒體時應同時考慮內容和情境。

### 2.10 Components of a Multimedia System / 多媒體系統的組成

**Source / 來源：L02 p. 22.**

| Component / 組件 | Function and examples / 功能及例子 |
|---|---|
| Capture devices / 擷取裝置 | Cameras, microphones, keyboards, mice, touchscreens, pressure sensors / 攝影機、麥克風、鍵盤、滑鼠、觸控屏及壓力感測器 |
| Storage / 儲存 | Preserve streams and metadata on disks, flash, optical media / 在磁碟、快閃記憶體及光碟保存資料流及中繼資料 |
| Networks / 網絡 | Carry data using local networks, cellular links, Bluetooth, radio / 經區域網絡、流動網絡、藍牙及無線電傳送資料 |
| Computing / 運算 | Workstations, servers, DSPs process and integrate data / 工作站、伺服器及數碼訊號處理器處理和整合資料 |
| Output / 輸出 | Displays, speakers, vibration motors, haptic devices / 顯示器、揚聲器、震動馬達及觸覺裝置 |

**Answer framework / 答題框架:** For any application, identify capture, representation/storage, communication, processing/integration, output, and feedback. Then discuss context, synchronization, and one likely failure. / 分析任何應用時，依次辨認擷取、表示／儲存、通訊、處理／整合、輸出和回饋，再討論情境、同步及一種可能故障。

<a id="lecture-03"></a>
## 3. Lecture 03: Example Applications / 應用實例

### 3.1 Audio Application Map / 聲音應用概覽

**Source / 來源：L03 pp. 2-21.**

| Application / 應用 | Input / 輸入 | Goal / 目標 | Main idea / 主要原理 |
|---|---|---|---|
| Noise cancellation / 噪音消除 | Sound and a noise estimate / 聲音及噪音估計 | Reduce unwanted sound / 減少不需要的聲音 | Subtraction or destructive interference / 相減或相消干涉 |
| Equalizer / 等化器 | Audio / 聲音 | Change tonal balance / 改變音色平衡 | Frequency-dependent gain / 按頻率調整增益 |
| Speech-to-text / 語音轉文字 | Speech waveform / 語音波形 | Recover words / 恢復詞語 | Acoustic patterns plus language constraints / 聲學圖樣加語言約束 |
| Shazam / 音樂辨識 | Short music recording / 短音樂錄音 | Identify a known recording / 識別已收錄的錄音 | Time-frequency fingerprints / 時頻指紋 |
| Audio compression / 聲音壓縮 | Speech or music / 語音或音樂 | Reduce bitrate / 減少位元率 | Production models or perceptual redundancy / 發聲模型或知覺冗餘 |

### 3.2 Noise Cancellation: Two Different Problems / 噪音消除的兩種問題

**Source / 來源：L03 pp. 3-7; links to L04 p. 40.**

**EN:** **Active noise cancellation (ANC)** in headphones or cars produces an acoustic anti-noise signal so the unwanted pressure wave is reduced at a target location through destructive interference. It requires the correct amplitude, phase, and timing; cancellation at one position does not guarantee cancellation everywhere.

**繁中：** 耳機或汽車中的 **主動降噪（ANC）** 會產生反噪音聲波，利用相消干涉，在目標位置減少不需要的聲壓波。反噪音必須有合適振幅、相位及時間；某個位置的抵消效果，不代表所有位置都同樣有效。

**EN:** **Microphone noise reduction** aims to clean the recorded signal. In the two-microphone slide, one microphone receives speech plus noise and another provides a noise reference. A simplified model is $x_1=s+n_1$, $x_2\approx n_2$, followed by subtracting a suitably adjusted estimate of $n_1$. The microphones do not generally receive identical noise, so literal unadjusted subtraction is only an idealized illustration.

**繁中：** **麥克風降噪** 的目的是清理錄音訊號。雙麥克風投影片中，一個接收語音加噪音，另一個提供噪音參考。簡化模型為 $x_1=s+n_1$、$x_2\approx n_2$，然後扣除經適當調整的 $n_1$ 估計。兩支麥克風一般不會接收到完全相同的噪音，因此不經調整直接相減只是理想化示意。

**EN:** More microphones can support spatial filtering and beamforming: signals arriving from one direction have characteristic relative delays and levels. Practical trade-offs include pickup distance, background noise, acoustic overload, size, and battery use. The slide's iPhone/Echo microphone counts are historical examples, not current product requirements.

**繁中：** 更多麥克風可支援空間濾波及波束成形：來自某方向的訊號，在不同麥克風之間具有特定時間差和音量差。實際設計須兼顧收音距離、背景噪音、聲學過載、體積及耗電。投影片的 iPhone／Echo 麥克風數量屬歷史例子，並非現今產品必須符合的規格。

### 3.3 Digital Audio Equalization / 數碼聲音等化

**Source / 來源：L03 p. 8.**

**EN:** An equalizer boosts or attenuates selected frequency bands. Bass controls affect lower frequencies; treble controls affect higher frequencies. The lecture gives an approximate audio range of 20 Hz-20 kHz, gain ranges such as ±6 dB to ±24 dB, and potentially more than 200 bands in high-end equipment. Presets apply a collection of band gains at once.

**繁中：** 等化器會提升或衰減指定頻帶。低音控制主要影響低頻，高音控制主要影響高頻。講義舉出約 20 Hz 至 20 kHz 的聲音範圍、例如 ±6 dB 至 ±24 dB 的增益範圍，以及高階設備可超過 200 個頻帶。預設模式一次套用一整組頻帶增益。

**EN:** Equalization changes spectral balance; it is not the same as raising every sample's level, nor does it universally separate speech from overlapping noise. Excessive boosting can exceed available headroom and cause clipping.

**繁中：** 等化改變的是頻譜平衡，不等於把所有樣本一律放大，也不能普遍地把頻率重疊的語音與噪音分開。過度提升頻帶可能超出餘裕而造成削波。

### 3.4 Speech-to-Text Pipeline / 語音轉文字流程

**Source / 來源：L03 p. 9; L05 pp. 9-15.**

```text
Analog speech / 類比語音
  -> A/D conversion / 類比轉數碼
  -> noise reduction and normalization / 降噪及正規化
  -> short-time segmentation / 短時分段
  -> acoustic features or phoneme evidence / 聲學特徵或音素證據
  -> statistical decoding with language constraints / 配合語言約束的統計解碼
  -> words / 詞語
```

**EN:** Short frames make a changing signal locally easier to analyse. Acoustic patterns support possible phonemes; statistical models resolve likely word sequences. English has roughly 40 phonemes, depending on dialect and analysis. A phoneme is a contrastive speech unit, not the same as a written letter.

**繁中：** 短時分段可把不斷變化的訊號變成局部較容易分析的片段。聲學圖樣提供可能的音素證據，再由統計模型決定較可能的詞序。英語約有 40 個音素，但會因口音及分析方式而異。音素是能區分語音意義的單位，不等於書面字母。

**Clarification / 釐清:** This is the lecture's classical explanatory pipeline. Modern end-to-end systems need not contain a separately exposed phoneme stage. / 這是講義用來解說的傳統流程；現代端到端系統不一定設有獨立、可見的音素處理階段。

### 3.5 Shazam: Audio Fingerprinting / Shazam：聲音指紋

**Source / 來源：L03 pp. 10-14.**

**EN:** Shazam identifies an existing recording from a short, possibly noisy sample. A **spectrogram** shows time on one axis, frequency on another, and energy or magnitude as brightness/colour. Stable, strong peaks form a sparse fingerprint that can be matched against a database. The lecture uses a roughly 10-second query sample as an example.

**繁中：** Shazam 利用短而可能含噪音的片段，識別資料庫中已有的錄音。**聲譜圖** 的一軸為時間，另一軸為頻率，再以亮度或顏色表示能量或幅度。穩定而明顯的峰值可組成稀疏指紋，與資料庫比對。講義以約 10 秒查詢片段作例子。

**Supplement: matching intuition / 補充：比對直覺。** Peak pairs can encode two frequencies and their time difference, such as $(f_1,f_2,\Delta t)$. Store the fingerprint with song identity and reference time. Matching many fingerprints with a consistent time offset provides stronger evidence than matching one isolated peak. Relative timing permits a query to start in the middle of a recording. / 峰值配對可編碼兩個頻率及時間差，例如 $(f_1,f_2,\Delta t)$，並連同歌曲身份及參考時間儲存。大量指紋若具有一致時間偏移，證據就比單一峰值吻合更強。相對時間表示法讓查詢片段即使由歌曲中段開始，仍可比對。

**EN:** Peak-based matching can tolerate some noise, reverberation, equalization, and overlapping sounds. Robustness is not immunity: severe distortion, insufficient matching peaks, or an unindexed recording can cause failure. This is recording identification, not necessarily lyric recognition or understanding a song's meaning.

**繁中：** 峰值比對可容忍一定程度的噪音、混響、等化及重疊聲音，但穩健不等於完全免疫。嚴重失真、峰值不足或資料庫沒有該錄音，都可能造成失敗。這是錄音身份辨識，不一定涉及歌詞辨認或理解歌曲意思。

### 3.6 LPC Vocoder: Modelling Speech Production / LPC 聲碼器：模擬語音產生

**Source / 來源：L03 pp. 15-16.**

**EN:** Linear Predictive Coding (LPC) represents speech using a simplified production model: an excitation source drives a filter approximating the vocal tract. Voiced speech has approximately periodic excitation related to vocal-fold vibration; unvoiced speech can be represented by noise-like excitation. The changing vocal tract shapes the spectral resonances.

**繁中：** 線性預測編碼（LPC）利用簡化發聲模型表示語音：激勵源驅動一個近似聲道的濾波器。有聲語音的激勵近似週期性，與聲帶振動有關；無聲語音則可用類噪音激勵表示。聲道形狀改變會塑造頻譜共振。

**EN:** Instead of every waveform sample, transmit parameters such as filter coefficients, energy, pitch, and voiced/unvoiced state. A decoder synthesizes approximate speech from them. The lecture links this framework to GSM voice coding. Another use is speech synthesis or controlled voice modification.

**繁中：** 系統不必傳送每個波形樣本，而可傳送濾波器係數、能量、基音，以及有聲／無聲狀態等參數。解碼器再利用這些參數合成近似語音。講義把這套架構連結到 GSM 語音編碼；其他用途包括語音合成或可控制的聲音改造。

**Exam distinction / 答題區分:** LPC exploits a model of how speech is produced; MPEG audio mainly exploits what listeners can perceive. A speech-specific model is not automatically suitable for arbitrary music. / LPC 主要利用語音如何產生的模型；MPEG 聲音主要利用人耳能感知甚麼。適用於語音的模型不一定適用於任意音樂。

### 3.7 MPEG Audio and Psychoacoustic Masking / MPEG 聲音與心理聲學遮蔽

**Source / 來源：L03 pp. 17-20.**

**EN:** MPEG perceptual audio coding is lossy. It reduces data allocated to components predicted to be less audible under a psychoacoustic model. The ear performs frequency-selective analysis through the cochlea; perception is not equally sensitive to every spectral component.

**繁中：** MPEG 知覺聲音編碼屬有損壓縮。它根據心理聲學模型，減少分配給預期較不易被聽見成分的資料。耳蝸具有頻率選擇性，因此人耳對不同頻譜成分並非同樣敏感。

| Masking / 遮蔽 | Meaning / 意義 | Coding implication / 編碼意義 |
|---|---|---|
| Frequency masking / 頻率遮蔽 | A strong component makes a weaker component in a nearby frequency region less audible / 強成分使鄰近頻率區域的弱成分較難被聽見 | Use fewer bits where resulting distortion remains below the masking threshold / 在失真仍低於遮蔽門檻處使用較少位元 |
| Temporal masking / 時間遮蔽 | A strong sound reduces audibility of a weak sound close to it in time; the lecture emphasizes the period after it / 強聲使時間上鄰近的弱聲較難被聽見；講義主要強調強聲停止後的短時間 | Account for time-varying audibility / 考慮隨時間變化的可聽程度 |

**EN:** The lecture distinguishes Layer I's basic scheme, Layer II's improved bit allocation, and Layer III's hybrid filtering. Its MP3 outline groups 1152 samples, divides audio into 32 subbands, estimates masking, and uses variable-length coding for remaining coding redundancy. At 44.1 kHz, 1152 samples correspond to approximately 26.12 ms.

**繁中：** 講義區分 Layer I 的基本架構、Layer II 的改良位元分配，以及 Layer III 的混合濾波。MP3 的簡化流程把 1152 個樣本成組、分成 32 個子頻帶、估計遮蔽，再以可變長度編碼處理餘下編碼冗餘。在 44.1 kHz 下，1152 個樣本約等於 26.12 毫秒。

**Clarification / 釐清:** The 1152-sample description fits the lecture's MPEG-1 Layer III example; frame sizes vary across MPEG versions. MP3 is MPEG Audio Layer III, not MPEG-3. The layers are related coding designs, not three mandatory encoding passes or three files that must all be stored. / 1152 個樣本的描述適用於講義的 MPEG-1 Layer III 例子；不同 MPEG 版本的影格大小可以不同。MP3 指 MPEG Audio Layer III，並非 MPEG-3。三個 Layer 是相關編碼設計，不是必須依次執行的三次編碼，也不是必須一同儲存的三份檔案。

### 3.8 Image Enhancement and Multi-Frame Fusion / 影像增強與多幀融合

**Source / 來源：L03 pp. 22-25.**

**EN:** Enhancement makes images more useful or visually clearer. Night mode combines frames captured under differing exposure conditions: longer exposures gather more light but risk motion blur; shorter ones better preserve moving detail but may be noisy. Aligning and fusing frames can improve detail and noise performance. The slides' Deep Fusion example describes combining nine images on particular iPhone 11 models; it is a product-specific historical example.

**繁中：** 影像增強令影像更有用或視覺上更清晰。夜間模式融合不同曝光條件下拍攝的畫面：長曝光收集較多光，但容易產生動態模糊；短曝光較能保留移動細節，卻可能有較多噪點。對齊及融合多幀可改善細節和噪點表現。投影片提到特定 iPhone 11 型號把九張影像結合的 Deep Fusion，屬產品特定的歷史例子。

**Supplement / 補充:** Retinex-style enhancement is often explained by separating illumination from reflectance, conceptually $I\approx L\times R$. Adjusting illumination can reveal detail, but aggressive enhancement can amplify noise or create artifacts. Multi-frame fusion also risks ghosting if moving regions are misaligned. / Retinex 類增強通常用照明與反射率分離來解釋，概念上為 $I\approx L\times R$。調整照明可顯露細節，但過度增強可能放大噪點或製造偽影。多幀融合若沒有對齊移動區域，也可能出現鬼影。

### 3.9 Video Stabilization Versus Motion Deblurring / 影片防震與動態去模糊

**Source / 來源：L03 pp. 26-30.**

**EN:** Stabilization reduces unwanted frame-to-frame camera movement. Deblurring attempts to reverse the spatial smearing accumulated **during an exposure**. Stabilizing already blurred frames does not automatically recover their detail. Blur can arise from camera motion, object motion, or both; different parts of an object may move differently relative to the camera.

**繁中：** 防震減少畫面與畫面之間不必要的相機移動；去模糊則嘗試逆轉 **曝光期間** 累積的空間拖影。把已模糊的畫面穩定下來，不會自動恢復細節。模糊可由相機移動、物件移動或兩者共同造成；同一物件不同部分相對相機的速度亦可能不同。

**Diagram walkthrough / 圖解流程：L03 p. 29.**

```text
Blurred frames / 模糊畫面
  -> foreground extraction and mask / 前景抽取及遮罩
  -> segmentation and corner detection / 分割及角點偵測
  -> feature tracking with HBMA / 以投影片所示 HBMA 追蹤特徵
  -> motion estimation / 動作估計
  -> blur-kernel calculation / 模糊核計算
  -> deblurring / 去模糊
  -> restored result / 還原結果
```

**EN:** The foreground mask helps focus motion analysis on the object. Tracked corners provide movement evidence; the blur kernel describes how a sharp point spreads during exposure. A single global kernel may fail when motion is spatially varying. Recovery is limited by noise and information already lost.

**繁中：** 前景遮罩協助把動作分析集中於目標物件。追蹤角點可提供位移證據；模糊核描述清晰點在曝光期間如何擴散。若不同位置有不同動作，單一全域模糊核可能失效。噪點及已流失的資訊亦會限制還原效果。

### 3.10 Face Detection, Recognition, and Tracking / 人臉偵測、辨識與追蹤

**Source / 來源：L03 pp. 31-34.**

| Task / 任務 | Question answered / 回答的問題 |
|---|---|
| Detection / 偵測 | Is there a face, and where? / 有沒有人臉？在哪裏？ |
| Tracking / 追蹤 | Where does this same target move over time? / 同一目標隨時間移到哪裏？ |
| Recognition / 辨識 | Whose face is it? / 是誰的臉？ |
| Verification / 驗證，補充分類 | Does this face match a claimed identity? / 是否符合聲稱的身份？ |

**EN:** The lecture's Viola-Jones detector uses efficiently computed Haar-like contrast features and AdaBoost to combine weak classifiers into a stronger decision system. It scans positions and scales to find faces of different sizes. A simple feature might compare brightness in an eye region with surrounding regions; no single feature is sufficient for all faces.

**繁中：** 講義的 Viola-Jones 偵測器使用可高效率計算的 Haar 類明暗對比特徵，再透過 AdaBoost 把弱分類器組成較強的決策系統。系統掃描不同位置和尺度，以尋找大小不同的人臉。簡單特徵可以比較眼部區域與周圍的亮度，但單一特徵不足以判斷所有人臉。

**EN:** Model-based head tracking fits or follows a head representation over successive frames. A virtual tripwire can count people when tracked heads cross a line. Tracking reduces the risk of counting the same person in every frame, but occlusion and identity switches remain possible.

**繁中：** 模型式頭部追蹤會在連續畫面中擬合或跟隨頭部表示。追蹤到的頭部越過虛擬警戒線時，可用作人流計數。追蹤能減少每幀都重複計算同一人的問題，但仍可能受遮擋和身份錯換影響。

### 3.11 3D Face Reconstruction / 三維人臉重建

**Source / 來源：L03 pp. 35-46.**

**EN:** The lecture's motivation is pose mismatch: a frontal gallery image is harder to match against a profile query using the illustrated recognition approach. Reconstructing a 3D face allows a more frontal view to be generated for comparison. This addresses pose variation, not every possible recognition problem.

**繁中：** 講義的出發點是姿態不一致：以示範的辨識方法，正面資料庫照片較難與側面查詢照片比對。重建三維人臉後，可以產生較正面的視角再比較。這主要處理姿態變化，並非解決所有辨識問題。

1. **Facial features / 臉部特徵:** Use Active Appearance Models (AAMs) for frontal, half-profile, and profile appearances; choose the best fit by minimum residual error. / 使用正面、半側面及側面的主動外觀模型（AAM），以最小殘差選擇最吻合模型。
2. **3D geometry / 三維幾何:** Fit or modify the face mesh. The example has 6,292 vertices and 6,152 facets. / 擬合或調整人臉網格；例子有 6,292 個頂點及 6,152 個面。
3. **Pose / 姿態:** Estimate orientation relative to the camera. / 估計人臉相對相機的方向。
4. **Texture / 紋理:** Rebuild the 2D texture map and map it to the mesh; render, fill/interpolate missing areas, and postprocess. / 重建二維紋理貼圖並映射到網格，再渲染、填補／插值缺失區域，以及後處理。
5. **Comparison / 比較:** Render a normalized view and compare with frontal gallery faces. / 渲染標準化視角，與正面資料庫人臉比對。

**EN:** A mesh represents shape; a texture map represents surface appearance. Texture coordinates tell the renderer which texture position belongs to each mesh region. A hidden cheek is not directly observed: filling it uses assumptions and cannot guarantee recovery of the person's true appearance.

**繁中：** 網格表示形狀；紋理貼圖表示表面外觀；紋理座標告訴渲染器每個網格區域對應貼圖哪個位置。被遮住的臉頰並沒有被直接觀測，填補時必須依靠假設，不能保證還原真實外貌。

**Clarification / 釐清:** The slides' claim that recognition works well only on frontal faces describes the illustrated approach and historical motivation, not a universal limitation of every modern system. The listed mesh sizes are implementation details, not the definition of 3D reconstruction. / 投影片所指「人臉辨識只在正面效果良好」，應理解為示範方法及歷史背景，而非所有現代系統的普遍限制。網格的頂點和面數亦是實作細節，不是三維重建的定義。

### 3.12 Recognition Limitations and Face ID / 辨識限制與 Face ID

**Source / 來源：L03 pp. 47-49.**

**EN:** Appearance changes and deliberately designed patterns can mislead recognition. The slides include adversarial spectacles and the difficulty of identifying unfamiliar people after small appearance changes. A high similarity score is evidence from a model, not proof of identity.

**繁中：** 外觀改變及刻意設計的圖樣可能誤導辨識。投影片包括對抗式眼鏡，以及陌生人外貌略變後便較難被認出的例子。高相似度分數只是模型提供的證據，不是身份的絕對證明。

**Supplement / 補充:** The Face ID example connects to depth sensing: infrared illumination and a projected pattern can support facial geometry estimation rather than relying only on an ordinary colour image. Depth, appearance, and anti-spoofing are distinct considerations; depth sensing does not make a system infallible. / Face ID 的例子可連結到深度感測：紅外線照明及投射圖樣可協助估計臉部幾何，而不只依賴普通彩色影像。深度、外觀和防偽屬不同考慮；有深度感測不代表系統絕不出錯。

### 3.13 Human Gait Recognition / 人體步態辨識

**Source / 來源：L03 pp. 50-54.**

**EN:** Gait is a behavioural biometric based on walking patterns. It can be useful at distances where facial detail is too weak. A silhouette removes much of the clothing texture and background while preserving coarse body shape and pose. Low resolution is still a limitation: enough temporal and shape evidence must remain.

**繁中：** 步態是根據行走模式的行為生物特徵，適合用於距離較遠、臉部細節不足的情況。輪廓圖可去除不少衣服紋理和背景，同時保留粗略身形和姿勢。不過低解像度仍有限制，必須保留足夠時間及形狀證據。

**Diagram walkthrough / 圖解：L03 pp. 52-53.** A dictionary contains example silhouette codewords. The shown input is assigned to the second of six codewords, giving the one-hot vector $[0,1,0,0,0,0]$. **Supplement:** Accumulating assignments across a sequence can form a codeword-frequency representation, compared with stored gait patterns. Such aggregation may lose temporal order unless order is separately represented. / 字典內包含輪廓碼字。投影片把輸入指派到六個碼字中的第二個，形成獨熱向量 $[0,1,0,0,0,0]$。**補充：** 把一段序列中的指派累積起來，可形成碼字頻率表示，再與已儲存步態比較；但若沒有另行保留次序，彙總可能流失時間順序。

**EN:** Viewpoint, clothing, carried objects, walking speed, injury, and occlusion can change gait evidence. Gait is a useful complementary cue, not a guaranteed unique identifier under all conditions.

**繁中：** 視角、衣着、攜帶物品、步速、受傷及遮擋都會改變步態證據。步態是有用的補充線索，但並非在所有條件下都能保證唯一身份。

### 3.14 Video Retrieval and Indexing / 影片檢索與索引

**Source / 來源：L03 pp. 55-59.**

```text
Video / 影片
  -> encoding and/or analysis / 編碼及／或分析
  -> motion cues + background model / 動作線索 + 背景模型
  -> moving-object segmentation / 移動物件分割
  -> tracking / 追蹤
  -> per-object feature extraction / 逐個物件抽取特徵
  -> indexed database / 索引資料庫
  -> retrieval request and matching / 檢索請求及比對
```

**EN:** Storing compressed video is different from making it searchable. Indexing connects raw video to metadata, analytics, and ultimately insights. The lecture uses colour distributions and edge orientations as examples of searchable features. Their combination can narrow candidates better than one feature alone.

**繁中：** 儲存壓縮影片，與令影片可以被搜尋，是兩回事。索引把原始影片連結到中繼資料、分析結果，最後產生洞見。講義用顏色分布及邊緣方向作可搜尋特徵；兩者結合通常比單一特徵更能縮窄候選結果。

**EN:** Colour histograms summarize colour but largely ignore arrangement. Edge orientation features summarize structure but do not directly encode identity. Compression motion vectors can provide motion cues, but their coding purpose means they are not guaranteed physical object trajectories. A request such as 'find the red car turning left' still requires bridging the semantic gap.

**繁中：** 顏色直方圖概括顏色，卻大致忽略空間排列；邊緣方向特徵概括結構，卻不直接代表身份。壓縮動作向量可提供動作線索，但其目的在於編碼，未必等於物件真實軌跡。「找出左轉的紅色車」這類請求，仍需要跨越語意鴻溝。

### 3.15 Metaverse, DIKW, Analytics, and Smart Cities / 元宇宙、DIKW、分析與智慧城市

**Source / 來源：L03 pp. 60-63.**

**EN:** The metaverse conversation example illustrates an experiential application: capture appearance and speech, reconstruct/render a representation, and communicate with a sense of presence. It depends jointly on capture, modelling, communication, synchronization, and reproduction.

**繁中：** 元宇宙對話例子展示體驗式應用：擷取外觀和語音、重建／渲染人物表示，再以具臨場感的方式溝通。它同時依賴擷取、建模、通訊、同步及重現。

| DIKW level / 層次 | Meaning / 意義 | Traffic example / 交通例子 |
|---|---|---|
| Data / 資料 | Raw observations / 原始觀測 | Video frames, GPS readings / 影片畫面、GPS 讀數 |
| Information / 資訊 | Organized data with context / 有情境的整理資料 | Vehicle count at a junction in five minutes / 路口五分鐘內車流量 |
| Knowledge / 知識 | Understood patterns and relationships / 已理解的模式及關係 | A recurring queue predicts congestion / 某種持續車龍模式預示擠塞 |
| Wisdom / 智慧 | Judgement supporting suitable action / 支援合適行動的判斷 | Decide whether and how to reroute traffic / 決定是否及如何改道 |

**EN:** As a teaching organization of the analytics examples: descriptive analytics asks what happened; predictive analytics estimates what may happen; prescriptive analytics considers what action to take. Data acquisition, cleaning, tagging, and correlation support these stages. A smart city combines traffic, environment, infrastructure, and other streams; more sensors alone do not create wisdom.

**繁中：** 可把講義的分析例子整理為：描述式分析問「發生了甚麼」；預測式分析估計「可能發生甚麼」；處方式分析考慮「應採取甚麼行動」。資料擷取、清理、標記及關聯分析支援這些階段。智慧城市整合交通、環境、基建等資料流，但增加感測器不會自動產生智慧。

### 3.16 Compare the Applications / 應用之間的比較

**EN:** Shazam and speech recognition both analyse time-frequency structure, but Shazam matches a known recording while speech recognition decodes words. LPC and MP3 both compress sound, but LPC models speech production while MP3 models perceptual audibility. Face recognition and gait recognition both support identity inference, but use different spatial/temporal evidence. Video retrieval brings these features into a search system rather than merely replaying video.

**繁中：** Shazam 和語音辨識都分析時頻結構，但前者比對已知錄音，後者解碼詞語。LPC 和 MP3 都壓縮聲音，但前者模擬語音產生，後者模擬知覺上的可聽程度。人臉和步態辨識都支援身份推斷，但所用的空間／時間證據不同。影片檢索則把這些特徵帶入搜尋系統，而不只是重播影片。

<a id="lecture-04"></a>
## 4. Lecture 04: Sensors, Digitization, and Sound / 感測器、數碼化與聲音

### 4.1 What Is a Sensor? / 甚麼是感測器？

**Source / 來源：L04 pp. 2-4.**

**EN:** A sensor measures a physical quantity and converts it into a signal usable by humans or machines. The lecture groups sensors into those resembling human senses, those measuring life/health, and those measuring environmental facts. The groups are practical categories and can overlap. Sound and light sensors are especially important to multimedia.

**繁中：** 感測器量度物理量，並把它轉成供人或機器使用的訊號。講義把感測器分為模擬人類感官、量度生命／健康，以及量度環境物理事實三類；這些是實用分類，彼此可以重疊。聲音及光感測器對多媒體尤其重要。

### 4.2 Touch Sensors and Actuators / 觸覺感測器與致動器

**Source / 來源：L04 pp. 5-7.**

**EN:** Skin is the human touch organ. Tactile sensors convert contact or pressure into signals. A resistive touchscreen detects electrical changes when layers contact under pressure; a capacitive touchscreen detects capacitance changes caused by touch. Pressure-sensitive input, such as the historical 3D Touch example, adds force-related information beyond contact position.

**繁中：** 皮膚是人類的觸覺器官。觸覺感測器把接觸或壓力轉成訊號。電阻式觸控屏透過受壓後接觸的層之間產生電氣變化來偵測；電容式觸控屏則偵測觸碰引起的電容量變化。講義以歷史性的 3D Touch 為例，說明壓力感應可在位置以外增加力度資訊。

**EN:** A sensor measures; an actuator acts. A vibration motor or Taptic Engine creates tactile output and is therefore an actuator, not the touch sensor itself. Together they form an interaction loop: press -> measurement -> computation -> haptic response.

**繁中：** 感測器負責量度，致動器負責產生作用。震動馬達或 Taptic Engine 製造觸覺輸出，因此是致動器，不是觸控感測器本身。兩者可形成互動循環：按壓 -> 量度 -> 運算 -> 觸覺回應。

### 4.3 Sight and Binocular Vision / 視覺與雙眼視覺

**Source / 來源：L04 pp. 8-15.**

**EN:** Cameras use photoelectric sensing elements, commonly CCD or CMOS arrays, to convert light into electrical measurements. More sensing elements generally support higher nominal pixel resolution, but actual detail also depends on optics, noise, focus, exposure, and processing.

**繁中：** 攝影機使用光電感測元件，常見為 CCD 或 CMOS 陣列，把光轉為電氣量度。較多感光元件通常支援較高標稱像素解像度，但真實細節仍受鏡頭、噪點、對焦、曝光和處理影響。

**EN:** The spot-the-difference demonstrations involve object identification, spatial relationships, working memory, and action, illustrating that seeing is more than recording pixels. Binocular vision compares different views from two eyes; stereograms exploit controlled disparities to create depth perception from a flat image.

**繁中：** 找不同遊戲涉及物件識別、空間關係、工作記憶及動作，說明「看見」不只是記錄像素。雙眼視覺比較兩眼不同視角；立體圖則利用刻意安排的視差，令平面影像產生深度知覺。

### 4.4 Hearing, Smell, and Taste / 聽覺、嗅覺與味覺

**Source / 來源：L04 pp. 16-19.**

**EN:** A microphone converts changing air pressure into an electrical signal. A dynamic microphone commonly uses a diaphragm/coil moving in a magnetic field; a condenser microphone senses capacitance changes between a moving diaphragm and a fixed electrode. Smell and taste involve chemical sensing and are harder to capture and reproduce in general-purpose multimedia systems.

**繁中：** 麥克風把空氣壓力變化轉成電訊號。動圈式麥克風一般利用振膜／線圈在磁場中移動；電容式麥克風則感應活動振膜與固定電極之間的電容量變化。嗅覺和味覺涉及化學感測，較難在通用多媒體系統中擷取和重現。

**Clarification / 釐清:** The slide's absolute statement that artificial noses/tongues have not been produced is too broad. Electronic noses and tongues exist for particular classification tasks; general reproduction of the complete human experience remains difficult. The course focuses primarily on sight and hearing. / 投影片所指「尚未製造出人工鼻或舌」過於絕對。電子鼻及電子舌已可處理特定分類任務，但全面重現人類感受仍然困難。本課主要集中視覺和聽覺。

### 4.5 Life and Environmental Sensors / 生命及環境感測器

**Source / 來源：L04 pp. 20-22.**

| Sensor / 感測器 | Directly measures or estimates / 直接量度或估計 | Important distinction / 重要區分 |
|---|---|---|
| Weight sensor / 體重感測器 | Load, converted to body weight / 負載，再換算體重 | BMI also needs height / BMI 亦需要身高 |
| Temperature sensor / 溫度感測器 | Temperature / 溫度 | Temperature alone is not a diagnosis / 單靠體溫並非診斷 |
| Heart-rate sensor / 心率感測器 | Pulse-related signal and rate / 脈搏相關訊號及速率 | Heart rate differs from ECG waveform / 心率不同於心電圖波形 |
| ECG sensor / 心電感測器 | Cardiac electrical activity / 心臟電活動 | Rhythm interpretation requires processing / 心律判讀須經處理 |
| Noise sensor / 噪音感測器 | Acoustic level / 聲音水平 | Weighting and calibration affect readings / 加權及校準影響讀數 |
| Accelerometer / 加速度計 | Acceleration-related measurement / 與加速度相關的量度 | Supports step detection and gravity-based tilt estimates / 可支援計步及利用重力估計傾斜 |
| Gyroscope / 陀螺儀 | Angular velocity / 角速度 | Rotation is not geographic location / 旋轉不等於地理位置 |
| GPS receiver / GPS 接收器 | Position and related timing estimates / 位置及相關時間估計 | Signals may be weak indoors / 室內訊號可能較弱 |
| Radiation sensor / 輻射感測器 | Radiation / 輻射 | Measures quantities beyond ordinary perception / 可量度一般感官不能直接感知的量 |
| Altimeter / 高度計 | Altitude estimate / 高度估計 | Some types infer altitude from air pressure / 部分類型從氣壓推算高度 |

**EN:** Combining movement and heart rate may support an estimate of energy expenditure, not a direct measurement of calories. The elderly-care example illustrates why combining health, activity, and communication sensors is useful. Instruments can also be more reliable than human perception in situations such as spatial disorientation.

**繁中：** 結合動作和心率可以支援熱量消耗估計，但並非直接量度卡路里。長者照護例子說明整合健康、活動及通訊感測器的價值。在空間迷向等情況下，儀器亦可能比人的感覺更可靠。

**Clarification / 釐清:** Standard BMI is $\text{mass in kg}/(\text{height in m})^2$. Age and sex may affect interpretation or other body-composition estimates, but they are not inputs to that basic formula. / 標準 BMI 為「公斤體重 ÷ 公尺身高的平方」。年齡及性別可影響解讀或其他身體組成估計，但不屬於這條基本公式的輸入。

### 4.6 Sensor Limitations / 感測器限制

**Source / 來源：L04 pp. 23-24.**

| Limitation / 限制 | Definition / 定義 | Example / 例子 |
|---|---|---|
| Limited dynamic range / 動態範圍有限 | Only a finite interval of input levels can be represented usefully / 只有有限輸入水平能被有效表示 | Retain shadow detail but saturate highlights, or preserve highlights but lose shadows / 保留暗位卻令亮位飽和，或保留亮位卻失去暗位 |
| Offset or bias / 偏移或偏差 | Nonzero output for zero input / 輸入為零時輸出不為零 | A stationary measurement has a fixed baseline error / 靜態量度仍有固定基線誤差 |
| Nonlinearity / 非線性 | Output does not change proportionally with input / 輸出不隨輸入成比例改變 | Doubling input does not double the useful response / 輸入加倍，反應卻不加倍 |
| Hysteresis / 遲滯 | Output at the same input depends on previous input history / 相同輸入的輸出取決於先前歷程 | Increasing and decreasing pressure follow different curves / 加壓和減壓走不同曲線 |
| Drift / 漂移，釐清用語 | Response changes over time under otherwise similar conditions / 其他條件相若時，反應仍隨時間改變 | Baseline shifts as equipment warms up / 設備升溫時基線改變 |

**Clarification / 釐清:** The slide uses a time-deviation phrase under hysteresis, but its diagram specifically shows history-dependent increasing/decreasing paths. Do not equate hysteresis with ordinary drift. Calibration can reduce some bias or nonlinearity, but cannot recreate information lost through saturation. / 投影片在遲滯下使用了隨時間偏離的字眼，但圖中實際顯示輸入上升及下降時的歷程相依。不要把遲滯等同一般漂移。校準可減少部分偏差或非線性，卻不能重建飽和後已流失的資訊。

### 4.7 Signals and Sinusoids / 訊號與正弦波

**Source / 來源：L04 pp. 25-30.**

**EN:** A sinusoid can be visualized as the projection of a point moving uniformly around a circle. It is a useful building block for understanding complex signals, just as simple pieces can build a larger structure. Write

**繁中：** 正弦波可視為圓周上等速移動點的投影。它是理解複雜訊號的重要基本構件，就像簡單積木能組成大型結構。可寫成：

$$
x(t)=A\sin(2\pi ft+\phi),\qquad T=\frac{1}{f},\qquad \omega=2\pi f.
$$

| Symbol / 符號 | Meaning / 意義 |
|---|---|
| $A$ | Amplitude; units depend on the measured quantity / 振幅；單位取決於量度對象 |
| $f$ | Frequency in hertz, cycles per second / 頻率，單位 Hz，即每秒週期數 |
| $T$ | Period in seconds / 週期，單位秒 |
| $\phi$ | Phase in radians / 相位，單位弧度 |
| $\omega$ | Angular frequency in radians per second / 角頻率，單位弧度每秒 |

**EN:** Increasing amplitude is not increasing frequency. A time shift changes phase. For waves of the same frequency, phase differences determine whether their sum reinforces or cancels them.

**繁中：** 增加振幅不等於增加頻率。時間平移會改變相位。相同頻率的波相加時，會增強還是抵消，取決於相位差。

### 4.8 Digitization = Sampling + Quantization / 數碼化 = 取樣 + 量化

**Source / 來源：L04 pp. 31-33.**

**EN:** An analogue electrical signal varies continuously. **Sampling** discretizes time; **quantization** discretizes amplitude. They control different aspects of fidelity. Sampling alone does not force amplitude to have only a finite number of values.

**繁中：** 類比電訊號是連續變化的。**取樣** 把時間離散化；**量化** 把振幅離散化。兩者控制不同層面的保真度。單做取樣，並不會令振幅只能取有限數目的值。

$$
T_s=\frac{1}{f_s},\qquad x[n]=x(nT_s),\qquad L=2^b.
$$

**EN:** Here $f_s$ is sampling rate, $T_s$ is sample interval, $n$ is the integer sample index, $b$ is bits per sample, and $L$ is the number of quantization levels. With 8 bits there are 256 levels; with 16 bits there are 65,536. More bits give finer amplitude resolution **for the same full-scale range**.

**繁中：** $f_s$ 是取樣率，$T_s$ 是取樣間隔，$n$ 是整數樣本索引，$b$ 是每樣本位元數，$L$ 是量化級數。8 位元有 256 級，16 位元有 65,536 級。**滿刻度範圍相同時**，更多位元代表更細緻的振幅解像度。

**Supplement / 補充:** For an ideal uniform quantizer with total span $R$ and $2^b$ equal-width bins, $\Delta=R/2^b$. Rounding to the nearest bin midpoint gives an error magnitude no greater than approximately $\Delta/2$ for non-overloaded inputs. Endpoint conventions can change the step definition; state the assumed quantizer. / 理想均勻量化器若總範圍為 $R$，並分成 $2^b$ 個等寬區間，則 $\Delta=R/2^b$。對未超出量程的輸入，四捨五入到最近區間中心時，誤差絕對值一般不超過約 $\Delta/2$。不同端點定義會改變步距寫法，因此應交代量化器假設。

**Exam trap / 常見陷阱:** Raising the bit depth cannot fix aliasing caused by insufficient sampling; raising sampling rate does not by itself remove coarse amplitude quantization. / 增加位元深度不能修正取樣不足造成的混疊；提高取樣率本身亦不能消除粗糙振幅量化。

### 4.9 Nyquist Frequency and Aliasing / 奈奎斯特頻率與混疊

**Source / 來源：L04 pp. 33-34, 49.**

**EN:** If a baseband signal has no frequency components above $B$, ideal reconstruction requires sampling sufficiently fast. The lecture emphasizes the safe condition

**繁中：** 若基頻訊號不含高於 $B$ 的頻率成分，理想重建需要足夠高的取樣率。講義強調較穩妥的條件：

$$
f_s>2B,\qquad f_N=\frac{f_s}{2}.
$$

**EN:** $2B$ is the Nyquist rate associated with the signal bandwidth; $f_N$ is the Nyquist frequency associated with a chosen sampler. **Aliasing** occurs when different analogue frequencies produce indistinguishable samples. Information is then ambiguous even if stored samples are numerically precise.

**繁中：** $2B$ 是與訊號頻寬相關的奈奎斯特取樣率；$f_N$ 是與既定取樣器相關的奈奎斯特頻率。**混疊** 指不同類比頻率產生無法區分的樣本。即使儲存的數值非常精確，資訊本身仍然有歧義。

**Why strict inequality? / 為何用嚴格大於？** Let $x(t)=A\sin(2\pi Bt)$. At $f_s=2B$, $x[n]=A\sin(\pi n)=0$ for every integer $n$: a nonzero sinusoid can look like silence. More generally, sampling $A\cos(2\pi Bt+\phi)$ at that boundary loses the sine-phase component. This explains the phase ambiguity illustrated in the slide. / 設 $x(t)=A\sin(2\pi Bt)$，在 $f_s=2B$ 時，所有整數 $n$ 都有 $x[n]=A\sin(\pi n)=0$，非零正弦波可能看來像靜音。更一般地，在該邊界取樣 $A\cos(2\pi Bt+\phi)$ 會失去正弦相位成分，正好解釋投影片的相位歧義。

**Supplement / 補充:** Use an analogue anti-aliasing low-pass filter **before** sampling, and leave a practical transition band. Once frequencies have aliased into the same sampled sequence, an ordinary digital filter cannot generally distinguish their origins. / 應在取樣 **之前** 使用類比抗混疊低通濾波器，並預留實際濾波器的過渡頻帶。不同頻率一旦混疊成同一樣本序列，普通數碼濾波器一般不能再分辨其來源。

**Audio example / 聲音例子:** At 44.1 kHz, the Nyquist frequency is 22.05 kHz. A 20 kHz audio band leaves some filter transition space. If the desired upper limit were exactly 22 kHz, sampling at exactly 44 kHz would be the problematic boundary, not generous headroom. / 44.1 kHz 取樣的奈奎斯特頻率為 22.05 kHz，因此 20 kHz 聲音頻帶仍有一些濾波過渡空間。若要保留至 22 kHz，恰好 44 kHz 取樣正是有問題的邊界，並沒有充裕餘量。

### 4.10 Physical Properties of Sound / 聲音的物理性質

**Source / 來源：L04 p. 35.**

**EN:** In air and water, sound propagates as mechanical pressure variations, commonly described as longitudinal waves. It needs a medium and does not travel through a vacuum. The lecture gives approximately 343 m/s in dry air at 20°C and 1,482 m/s in water at 20°C. Speed depends on the medium and conditions.

**繁中：** 在空氣和水中，聲音以機械壓力變化傳播，通常描述為縱波。它需要介質，不能在真空中傳播。講義給出 20°C 乾燥空氣中約 343 m/s，以及 20°C 水中約 1,482 m/s；速度取決於介質及條件。

**Supplement / 補充:** $v=f\lambda$, where $v$ is propagation speed and $\lambda$ wavelength. In the same medium, higher frequency means shorter wavelength. / $v=f\lambda$，其中 $v$ 為傳播速度，$\lambda$ 為波長。在同一介質中，頻率愈高，波長愈短。

**Clarification / 釐清:** The slides give slightly different hearing ranges: about 12 Hz-22 kHz on L04 p. 35, 20 Hz-22 kHz on p. 49, and 20 Hz-20 kHz in the equalizer discussion. Treat these as approximate examples; the conventional teaching range is about 20 Hz-20 kHz for young healthy listeners, with substantial individual variation. Use the range explicitly specified in a calculation question. / 講義的聽覺範圍略有不同：L04 第 35 頁約為 12 Hz 至 22 kHz，第 49 頁為 20 Hz 至 22 kHz，等化器部分則為 20 Hz 至 20 kHz。應視為近似例子；一般教學常用健康年輕人的約 20 Hz 至 20 kHz，但個體差異很大。計算題應採用題目明確指定的範圍。

### 4.11 Sound Pressure Level / 聲壓級

**Source / 來源：L04 p. 36.**

$$
L_p=10\log_{10}\left(\frac{p_{\mathrm{rms}}^2}{p_{\mathrm{ref}}^2}\right)
=20\log_{10}\left(\frac{p_{\mathrm{rms}}}{p_{\mathrm{ref}}}\right)\ \mathrm{dB}.
$$

**EN:** $p_{\mathrm{rms}}$ is RMS sound pressure, not peak pressure. In air use $p_{\mathrm{ref}}=20\ \mu\mathrm{Pa}$; the slide gives $1\ \mu\mathrm{Pa}$ for water. A decibel value is a logarithmic ratio and needs a reference to have physical meaning. Different references mean air and underwater dB values cannot be compared directly as though their zero were identical.

**繁中：** $p_{\mathrm{rms}}$ 是均方根聲壓，不是峰值聲壓。空氣中使用 $p_{\mathrm{ref}}=20\ \mu\mathrm{Pa}$；投影片給出的水中參考值為 $1\ \mu\mathrm{Pa}$。分貝是對數比值，必須有參考值才具有物理意義。參考值不同，因此不能把空氣及水中的 dB 直接當作相同零點比較。

**EN:** Doubling RMS pressure adds about 6.02 dB; multiplying it by ten adds 20 dB. For a power ratio, use $10\log_{10}$; for a pressure-amplitude ratio, use $20\log_{10}$. A reference level of 0 dB SPL is not zero pressure or absolute silence.

**繁中：** RMS 聲壓加倍約增加 6.02 dB；增至十倍則增加 20 dB。功率比使用 $10\log_{10}$，聲壓振幅比使用 $20\log_{10}$。0 dB SPL 是參考水平，並不代表聲壓為零或絕對無聲。

### 4.12 A-Weighting / A 加權

**Source / 來源：L04 p. 37.**

**EN:** Human hearing sensitivity depends on frequency. A-weighting applies a standardized frequency response to approximate aspects of this sensitivity; results are labelled dB(A) or dBA. A-weighted and unweighted levels need not match for the same sound, particularly if it contains strong low-frequency energy.

**繁中：** 人耳敏感度隨頻率改變。A 加權使用標準化頻率響應，近似反映部分聽覺特性；結果標示為 dB(A) 或 dBA。同一聲音的 A 加權及未加權聲級未必相同，尤其當低頻能量較強時。

**Clarification / 釐清:** One cannot generally convert broadband dB SPL to dBA using one universal number. Frequency content is needed. Band corrections can be added to individual band levels, but combining band energies requires logarithmic, not ordinary arithmetic, summation. / 一般不能把寬頻 dB SPL 加減某個固定數字便轉成 dBA，因為需要知道頻譜。各頻帶可分別加上加權修正值，但合併頻帶能量時須使用對數相加，而不是普通算術相加。

### 4.13 Echo, Reverberation, and Interference / 回聲、混響與干涉

**Source / 來源：L04 pp. 38-40.**

| Effect / 現象 | Mechanism / 機制 | Consequence / 後果 |
|---|---|---|
| Echo / 回聲 | A delayed reflection of a sound / 聲音的延遲反射 | A later copy can interfere with recognition / 較遲的副本可能干擾辨識 |
| Reverberation / 混響 | Many reflections build up and decay in an enclosure / 室內大量反射疊加並逐漸衰減 | Smears temporal and spectral evidence / 模糊時間及頻譜線索 |
| Interference / 干涉 | Waves superpose / 多個波疊加 | Reinforcement or cancellation / 增強或抵消 |

**EN:** The lecture gives about 0.1 s as a rough delay below which humans may not hear a separate echo; actual perception depends on the sound and conditions. A machine can still be affected by reflections that a person does not hear as distinct echoes. For a single reflecting wall, the reflected path is longer; in the simplest direct-back case, the extra round trip gives $d=v\tau/2$.

**繁中：** 講義以約 0.1 秒作為人耳未必能分辨獨立回聲的粗略延遲，實際情況取決於聲音和環境。即使人沒有聽見獨立回聲，機器仍可能受反射影響。對單一反射牆，反射路徑較長；在最簡單的直線來回情況，距離為 $d=v\tau/2$。

**EN:** ANC uses destructive interference, but acoustic matching must occur at the listener's location. Reverberation and changing paths make automatic processing more difficult.

**繁中：** ANC 利用相消干涉，但聲波必須在聆聽者位置適當配合。混響和不斷改變的傳播路徑，會增加自動處理難度。

### 4.14 Recording and Microphone Directionality / 錄音與麥克風指向性

**Source / 來源：L04 pp. 41-45.**

**EN:** Early mechanical recording engraved pressure-related motion into a rotating medium; playback converted the groove back into vibration. A music box instead stores a predefined note pattern mechanically, rather than recording an arbitrary incoming waveform. Modern microphones convert membrane movement into electrical changes.

**繁中：** 早期機械錄音把與聲壓有關的動作刻在旋轉媒體上，播放時再把刻紋轉成振動。音樂盒則以機械方式儲存預設音符模式，而不是錄下任意輸入波形。現代麥克風把振膜移動轉為電氣變化。

**EN:** Directionality describes sensitivity to sounds arriving from different angles. A polar pattern normally shows relative response by direction, and its shape can change with frequency. Choosing directionality can improve the desired-sound/background ratio before digital processing.

**繁中：** 指向性描述對不同入射角聲音的敏感度。極座標指向圖通常表示不同方向的相對響應，而且形狀會隨頻率改變。選擇適當指向性，可在數碼處理之前改善目標聲音與背景聲的比例。

| Pattern / 指向模式 | Main property / 主要特性 | Limitation / 限制 |
|---|---|---|
| Omnidirectional / 全指向 | Similar sensitivity in all directions / 各方向敏感度相近 | Picks up unwanted surrounding sound / 容易收錄周圍非目標聲音 |
| Cardioid / 心形 | Strong front response, rejection toward 180° / 前方敏感，180° 後方抑制較強 | Off-axis response is frequency-dependent / 離軸響應隨頻率改變 |
| Supercardioid / 超心形 | Narrower front lobe, with a rear lobe / 前瓣較窄，但有後瓣 | Rear sound is not completely rejected / 並非完全拒收後方聲音 |
| Shotgun / 槍式 | Highly directional, often using an interference tube / 高指向性，常用干涉管 | Performance depends on frequency and environment / 表現受頻率及環境影響 |

**Clarification / 釐清:** The slide gives roughly 120° rejection for supercardioid. Exact null angles depend on the pattern; an ideal supercardioid is often quoted near 125°. The important contrast is a narrower front lobe plus some rear pickup. Omni microphones are not intrinsically limited to noise-free studios; suitability depends on the recording purpose. / 投影片把超心形抑制角概括為約 120°；實際零點取決於模式，理想超心形常以接近 125° 描述。重要差異是前瓣較窄，但仍有後方收音。全指向麥克風亦並非只能用於無噪音錄音室，適合與否取決於錄音目的。

### 4.15 Frequency Response, Gain, and Clipping / 頻率響應、增益與削波

**Source / 來源：L04 pp. 46-47.**

**EN:** Frequency response describes how output changes with frequency for a given input. An ideally flat microphone does not overemphasize particular frequencies within its intended band. Linearity with sound pressure is a separate property; a microphone can be approximately flat yet overload at high pressure.

**繁中：** 頻率響應描述在既定輸入下，輸出如何隨頻率改變。理想平坦麥克風在指定頻帶內不會特別強調某些頻率。聲壓線性則是另一特性；麥克風即使頻響近似平坦，仍可在高聲壓時過載。

**EN:** Too much preamplifier gain exceeds the available range and clips peaks, producing distortion. Too little gain gives a weak signal relative to noise and quantizer resolution; small changes may map to the same code. Gain staging should use the available range without exceeding it. Turning down a clipped recording later does not restore the missing peak shape.

**繁中：** 前置放大增益過大會超出量程並削去峰值，產生失真。增益過小則令訊號相對噪音及量化解像度過弱，小變化可能映射到相同代碼。增益設定應善用量程但不超標。錄音已削波後，再調低音量也不能恢復流失的峰值形狀。

### 4.16 Digital Microphones and Sampling Choices / 數碼麥克風與取樣率選擇

**Source / 來源：L04 pp. 48-49.**

**EN:** The AES42 digital microphone example includes configurable gain, mute, low-cut filtering, and, where the device supports it, polar patterns. A low-cut filter suppresses low frequencies such as rumble, but it can also remove desired low-frequency content. The slide lists example cutoffs of 40, 80, and 120 Hz and gain up to +63 dB; these are example device capabilities.

**繁中：** AES42 數碼麥克風例子包括可設定增益、靜音、低切濾波，以及裝置支援時可選的指向模式。低切濾波可減少低頻隆隆聲，但亦可能刪去需要的低頻內容。投影片列出 40、80、120 Hz 截止頻率及最高 +63 dB 增益，屬示例裝置能力。

**Clarification / 釐清:** Digital transmission can reduce some analogue cable-interference problems, but the slide's 'no interference problem' is not an absolute guarantee. Acoustic noise, front-end overload, timing faults, and transmission errors are still possible. / 數碼傳輸可減少部分類比線路干擾，但投影片的「沒有干擾問題」不是絕對保證；聲學噪音、前端過載、時序錯誤及傳輸錯誤仍可能存在。

**EN:** Common recording rates include 44.1, 48, and 96 kHz. Higher sample rates provide a higher Nyquist limit and may be useful for particular processing or non-audible signals, but do not automatically mean better audible quality. The analogue front end, filtering, bit depth, and recording conditions still matter.

**繁中：** 常見錄音取樣率包括 44.1、48 及 96 kHz。較高取樣率提供較高奈奎斯特上限，可用於特定處理或非可聽訊號，但不會自動令可聽音質更好。類比前端、濾波、位元深度及錄音環境仍然重要。

<a id="lecture-05"></a>
## 5. Lecture 05: Reproduction, Speech, Music, Light, and Vision / 重現、語音、音樂、光與視覺

### 5.1 Loudspeakers / 揚聲器

**Source / 來源：L05 pp. 2-3.**

**EN:** A loudspeaker performs the reverse transduction of a microphone: an electrical signal drives mechanical vibration, which produces pressure waves in a medium. Speakers have directionality, and their radiation patterns vary with frequency. Therefore, identical electrical output does not imply identical sound at every listener position.

**繁中：** 揚聲器進行與麥克風相反的能量轉換：電訊號驅動機械振動，再於介質中產生聲壓波。揚聲器亦有指向性，而且輻射模式隨頻率改變。因此，即使電氣輸出相同，不同聆聽位置接收到的聲音也未必相同。

| Characteristic / 特性 | Meaning / 意義 | Do not confuse with / 不應混淆 |
|---|---|---|
| Rated power / 額定功率 | Input handling under specified test conditions / 指定測試條件下可承受的輸入 | Loudness or sound quality by itself / 不等於響度或音質 |
| Impedance / 阻抗 | Frequency-dependent electrical load seen by an amplifier / 放大器所見、隨頻率變化的電氣負載 | Acoustic volume / 不等於聲音大小 |
| Crossover frequencies / 分頻頻率 | Boundaries where bands are routed between drivers / 把頻帶分配給不同單元的交接範圍 | Sampling frequency / 不等於取樣率 |
| Frequency range / 頻率範圍 | Band the system can reproduce under a stated criterion / 按指定準則可重現的頻帶 | Guarantee of perfectly flat response / 不保證響應完全平坦 |

**Clarification / 釐清:** The slide describes rated power as a destruction limit; practical specifications distinguish continuous, programme, and peak conditions. A bigger wattage number alone is not a complete performance comparison. / 投影片把額定功率描述為損壞前上限；實際規格會區分連續、節目及峰值等條件。單看瓦數較大，並不足以全面比較表現。

### 5.2 Crossovers and Enclosures / 分頻器與音箱

**Source / 來源：L05 pp. 4-5.**

**EN:** A multi-driver speaker distributes frequency bands to suitable drivers: a woofer handles low frequencies, a tweeter handles high frequencies, and some systems add midrange drivers. A crossover performs the filtering. Low-frequency reproduction needs substantial air displacement; using a larger radiating area is helpful. It is not simply that 'large mass can only move slowly.'

**繁中：** 多單元揚聲器把頻帶分給合適單元：低音單元處理低頻，高音單元處理高頻，部分系統另有中音單元。分頻器負責濾波。重現低頻需要推動較多空氣，因此較大發聲面積有幫助；不能只簡化成「質量大便只能慢速移動」。

**EN:** The diaphragm's front and rear produce roughly opposite pressure variations at low frequencies. Without suitable separation, they can meet and cancel. An enclosure controls this interaction, especially at low frequencies, and can also affect resonance and response. The slide's primary point is preventing uncontrolled front/back interaction.

**繁中：** 振膜前後在低頻時產生大致相反的聲壓變化。如果沒有適當隔離，兩者可能相遇並互相抵消。音箱控制這種互動，尤其在低頻，亦會影響共振和響應。投影片重點是避免前後聲波不受控制地互相干涉。

### 5.3 Surround and Virtual Surround / 環繞聲與虛擬環繞聲

**Source / 來源：L05 pp. 6-8.**

**EN:** Surround sound uses spatially arranged channels to give the impression of sound around a listener. In common notation, 5.1 means five main channels plus one low-frequency effects channel; 7.1 has seven main channels plus LFE. The '.1' is a channel convention, not one tenth of a speaker. Physical deployment can be limited by space and wiring.

**繁中：** 環繞聲利用空間配置的聲道，營造聲音包圍聆聽者的感覺。常見寫法中，5.1 表示五個主要聲道加一個低頻效果聲道；7.1 則有七個主要聲道加 LFE。「.1」是聲道慣例，不是十分之一個揚聲器。實際安裝會受空間和佈線限制。

**EN:** Virtual surround reproduces selected localization cues with fewer speakers or headphones. The lecture mentions a two-front-speaker plus subwoofer configuration and a strip of small speakers using digital sound projection. Success depends on room acoustics, listener position, and cue accuracy; it is not identical to adding physical sources in every direction.

**繁中：** 虛擬環繞聲透過較少揚聲器或耳機重現部分定位線索。講義提到兩個前置揚聲器加低音單元，以及利用一排小揚聲器作數碼聲音投射。效果取決於房間聲學、聆聽位置和線索準確度；不等於真的在每個方向都放置聲源。

| Cue / 線索 | Meaning / 意義 | Role / 作用 |
|---|---|---|
| ILD: interaural level difference / 雙耳聲級差 | Different levels at the two ears / 兩耳接收音量不同 | Helps infer lateral direction / 幫助判斷左右方向 |
| ITD: interaural time difference / 雙耳時間差 | Different arrival times / 兩耳接收時間不同 | Helps infer lateral direction / 幫助判斷左右方向 |
| HRTF: head-related transfer function / 頭部相關傳遞函數 | Direction-dependent filtering caused by the head, torso, and outer ears / 頭部、軀幹及外耳造成、隨方向改變的濾波 | Provides spectral and other spatial cues / 提供頻譜及其他空間線索 |

**EN:** Pinna reflections are part of HRTF effects, but an HRTF describes more than the pinna alone. Individual anatomy matters, so a generic HRTF may produce weaker or ambiguous localization for some listeners.

**繁中：** 耳廓反射是 HRTF 效應的一部分，但 HRTF 不只描述耳廓。個體身體結構很重要，因此通用 HRTF 對部分聽者的定位效果可能較弱或含糊。

### 5.4 Human Speech Production / 人類語音產生

**Source / 來源：L05 pp. 9-10; L03 pp. 15-16.**

**EN:** Airflow from the lungs is shaped by the vocal folds and vocal tract. Vocal-fold vibration provides a fundamental frequency for voiced sounds; the mouth, tongue, lips, and nasal passages shape resonances and articulation. **Pitch** is associated with the fundamental frequency; the complete speech spectrum contains many other frequencies.

**繁中：** 肺部氣流經聲帶和聲道塑形。聲帶振動為有聲語音提供基頻；口腔、舌頭、嘴唇及鼻腔則塑造共振和發音。**音高** 與基頻有關，但完整語音頻譜還包含許多其他頻率。

**EN:** The slide gives illustrative pitch ranges of 120-160 Hz for adult men and 220-330 Hz for women/children, and an approximate normal-speech band of 80 Hz-5 kHz. These are examples, not fixed biological limits. Speech can contain higher-frequency energy, especially in fricatives, and varies greatly by speaker and sound.

**繁中：** 投影片舉出的基音範圍為成年男性約 120 至 160 Hz，女性／兒童約 220 至 330 Hz，並以約 80 Hz 至 5 kHz 描述一般語音頻帶。這些是例子，不是固定生理界限。語音可包含更高頻能量，尤其是摩擦音，而且會因說話者及語音種類而大幅改變。

| Speech class / 語音類別 | Airflow behaviour / 氣流特徵 | Examples / 例子 |
|---|---|---|
| Stops / 塞音 | Closure followed by release / 阻塞後釋放 | /p/, /t/, /k/ |
| Fricatives / 摩擦音 | Constriction creates turbulence / 收窄氣流產生湍流 | /f/, /v/, /s/ |
| Approximants / 近音 | Relatively open constriction, little turbulence / 收窄較輕，湍流較少 | /w/, /r/ |
| Aspirate / 送氣聲 | Breath-like airflow / 呼氣狀氣流 | /h/ |
| Vowels / 元音 | Relatively unobstructed airflow, typically voiced / 氣流較暢通，通常有聲 | Vowel sounds in words, not merely letter names / 詞中的元音，不只是字母名稱 |

**EN:** Phoneme inventories differ between languages. A written spelling is not a direct acoustic representation, which is one reason speech recognition requires language knowledge.

**繁中：** 不同語言有不同音素系統。文字拼寫並不是直接聲學表示，因此語音辨識需要語言知識。

### 5.5 Reading a Speech Spectrogram / 如何閱讀語音聲譜圖

**Source / 來源：L05 pp. 12-13.**

**EN:** A waveform primarily shows amplitude over time. A spectrum describes frequency content for a selected interval. A spectrogram shows how frequency content changes over time. Bright bands indicate stronger energy, but the scale and colour mapping must be checked.

**繁中：** 波形圖主要顯示振幅隨時間變化；頻譜顯示某一時段內的頻率成分；聲譜圖則顯示頻率成分如何隨時間改變。亮帶表示較強能量，但仍須留意座標刻度及顏色映射。

**EN:** Voiced sounds often show harmonic structure. **Formants** are vocal-tract resonance regions with concentrated energy; they are not the same as harmonics. Changing vocal-tract shape changes formants, helping distinguish vowels even when pitch is similar. Stops may show silence/closure followed by a broadband burst. Fricatives and aspirates often show more noise-like energy.

**繁中：** 有聲語音常呈現諧波結構。**共振峰（formants）** 是聲道共振造成的能量集中區域，不等於諧波。聲道形狀改變會改變共振峰，即使音高相近，也有助區分元音。塞音可能呈現靜音／閉塞後出現寬頻爆發；摩擦音及送氣聲則常有較類似噪音的能量。

**Clarification / 釐清:** The lecture contrasts voiced formant bands with unvoiced noise patterns. More precisely, unvoiced sounds lack periodic voiced excitation, but vocal-tract resonances can still shape their spectra. Do not equate 'no harmonic ladder' with 'no vocal-tract filtering.' / 講義以有聲共振峰帶與無聲噪音圖樣作對比。更精確地說，無聲語音缺乏週期性的有聲激勵，但聲道共振仍可塑造頻譜。不要把「沒有諧波階梯」等同「沒有聲道濾波」。

### 5.6 Speech Recognition Limitations / 語音辨識限制

**Source / 來源：L05 pp. 14-15.**

**EN:** Noise changes the observation and can also change how people speak: they may speak louder, alter pitch, or stretch syllables. Thus a recognizer trained in quiet conditions may face both acoustic noise and a changed speech-production pattern. Noise reduction alone does not eliminate all of this mismatch.

**繁中：** 噪音不只改變接收訊號，也可能改變人的說話方式：提高音量、改變音高，或拉長音節。因此，在安靜環境訓練的辨識器，同時面對聲學噪音及發聲模式改變。只做降噪不能消除全部不匹配。

**EN:** Sampling at 8 kHz gives a 4 kHz Nyquist limit, so higher-frequency speech cues are unavailable. This can make some consonants harder to distinguish, even though ordinary speech remains understandable. Spelling with distinctive whole words provides additional linguistic redundancy over a degraded channel.

**繁中：** 8 kHz 取樣的奈奎斯特上限為 4 kHz，因此更高頻語音線索無法保留。即使一般對話仍能理解，部分輔音亦可能較難區分。透過具辨識度的完整詞語拼讀，可為品質較差的通道增加語言冗餘。

**Clarification / 釐清:** Do not memorize 'all sibilants exceed 20 kHz' as a general rule. The key point is loss of useful high-frequency cues; /f/ is a fricative but not normally classified as a sibilant. Likewise, the slide's example spelling words illustrate the idea, not an authoritative standardized phonetic alphabet. / 不要把「所有噝音都超過 20 kHz」當成通則。重點是有用高頻線索可能流失；/f/ 是摩擦音，但通常不屬噝音。投影片的拼讀詞語亦只是示意，不是權威標準拼讀字母表。

### 5.7 Speech Synthesis, NLP, and Actions / 語音合成、自然語言處理與行動

**Source / 來源：L05 pp. 11, 15.**

**EN:** Speech recognition maps sound to a linguistic representation; text-to-speech maps text to generated speech. Natural language processing can interpret recognized words as an intent. Actionable intelligence uses that interpretation to select an operation. Recognizing 'turn off the kitchen light' is different from identifying the correct device and executing the command.

**繁中：** 語音辨識把聲音映射成語言表示；文字轉語音則把文字轉成合成聲音。自然語言處理可把辨識詞句解讀為意圖；可行動的智能再根據該解讀選擇操作。辨認出「關掉廚房燈」，與找出正確裝置並真正執行指令，是不同階段。

### 5.8 Music: Fundamental, Harmonics, and Overtones / 音樂：基頻、諧波與泛音

**Source / 來源：L05 p. 16.**

**EN:** Pitch is the perceptual attribute closely associated with fundamental frequency $f_0$. For a harmonic tone, the $n$th harmonic occurs at $nf_0$. At $f_0=440$ Hz, the first four harmonics are 440, 880, 1320, and 1760 Hz. The fundamental is the first harmonic; the first overtone is the next component above the fundamental, often the second harmonic in a harmonic series.

**繁中：** 音高是與基頻 $f_0$ 密切相關的知覺屬性。對諧和音而言，第 $n$ 次諧波位於 $nf_0$。若 $f_0=440$ Hz，前四個諧波為 440、880、1320 和 1760 Hz。基頻是第一諧波；第一泛音是基頻之上的下一個成分，在諧和系列中通常是第二諧波。

**Clarification / 釐清:** A partial is any component of a complex tone; not every partial must be an integer multiple of $f_0$. Some percussion instruments have strongly inharmonic partials. Pitch, loudness, and timbre describe different perceptual dimensions. / 分音是複合音中的任何成分，不一定都是 $f_0$ 的整數倍。部分敲擊樂器具有明顯非諧和分音。音高、響度和音色描述不同知覺維度。

### 5.9 Timbre and the ADSR Envelope / 音色與 ADSR 包絡

**Source / 來源：L05 p. 17.**

**EN:** Timbre is the quality that lets us distinguish sound sources even when they play the same note at comparable loudness. It depends on spectral structure and its evolution over time, including the amplitude envelope.

**繁中：** 音色讓我們即使在音符相同、響度相近時，仍能分辨不同聲源。它取決於頻譜結構及其時間變化，包括振幅包絡。

| Stage / 階段 | Meaning / 意義 |
|---|---|
| Attack / 起音 | Rise from onset toward a peak / 由開始上升至峰值 |
| Decay / 衰減 | Fall from the peak toward a sustained level / 由峰值下降至持續水平 |
| Sustain / 持續 | Level held while the note is maintained / 音符維持時的水平 |
| Release / 釋音 | Fall toward silence after excitation or the key is released / 激勵或按鍵停止後下降至靜音 |

**EN:** Sustain is usually a **level**, while attack, decay, and release are commonly specified using times. A piano note has a sharp attack and subsequent decay; a bowed string can maintain sound while excitation continues. A real instrument may be more complex than a four-stage model.

**繁中：** Sustain 通常是 **水平值**，而 attack、decay、release 常以時間表示。鋼琴音有明顯起音，之後逐步衰減；拉弦樂器則可在持續激勵下維持聲音。真實樂器可能比四階段模型複雜。

### 5.10 Instruments and Music Synthesis / 樂器與音樂合成

**Source / 來源：L05 pp. 18-20.**

**EN:** String instruments excite vibrating strings; wind instruments excite an air-column/resonator system; percussion instruments are struck, shaken, rubbed, or scraped. How they are excited affects harmonics and temporal envelopes.

**繁中：** 弦樂器激發琴弦振動；管樂器激發空氣柱／共鳴系統；敲擊樂器則透過敲、搖、摩擦或刮動發聲。激勵方式會影響諧波及時間包絡。

**EN:** The lecture presents modelling and sample/wavetable-based synthesis. A synthesizer can generate waveforms, combine tones or harmonics, and apply an ADSR envelope. Physical modelling simulates aspects of the vibrating system; sample-based methods reuse recorded material. Merely adding sinusoids is more specifically additive synthesis, although it serves the lecture's broader modelling explanation. Synthetic singing, illustrated by Hatsune Miku, connects synthesis with language and performance control.

**繁中：** 講義介紹模型式及樣本／波表式合成。合成器可產生波形、組合音調或諧波，再套用 ADSR 包絡。物理建模模擬振動系統的部分機制；樣本式方法則重用錄製素材。單純疊加正弦波更精確地稱為加法合成，但可用於理解講義廣義的建模說明。初音未來的合成歌聲例子，則連結到語言和演唱控制。

### 5.11 Light as an Electromagnetic Wave / 光是一種電磁波

**Source / 來源：L05 pp. 21-23.**

**EN:** The lecture uses approximately 400-780 nm for visible wavelengths. In a vacuum, light travels at exactly 299,792,458 m/s, usually approximated as $3.00\times10^8$ m/s. Frequency and wavelength are related by $c=f\lambda$ in a vacuum. Unlike sound, light does not require a material medium.

**繁中：** 講義以約 400 至 780 nm 表示可見光波長。光在真空中的速度精確為 299,792,458 m/s，通常近似為 $3.00\times10^8$ m/s。真空中頻率和波長的關係為 $c=f\lambda$。與聲音不同，光不需要物質介質。

**EN:** **Coherence** concerns stable phase relationships. Ordinary broad light sources are generally not mutually phase-coherent. **Polarization** concerns the orientation behaviour of the electric field. Unpolarized light has no fixed preferred orientation over the relevant observation; a polarizing filter selects components aligned with its transmission direction.

**繁中：** **相干性** 關乎穩定相位關係；普通寬頻光源一般沒有彼此穩定的相位關係。**偏振** 關乎電場方向的行為。非偏振光在相關觀測尺度下沒有固定優先方向；偏振片會選取沿其透射方向的成分。

**Exam distinction / 答題區分:** Coherence and polarization are not synonyms. Polarizing a source does not by itself make all its waves phase-coherent. / 相干和偏振不是同義詞。把光偏振化，不會自動令所有光波具有穩定相位關係。

### 5.12 Candela, Lumen, Lux, and Solid Angle / 坎德拉、流明、勒克斯與立體角

**Source / 來源：L05 pp. 24-29.**

| Quantity / 物理量 | Unit / 單位 | Meaning / 意義 |
|---|---|---|
| Luminous intensity $I_v$ / 發光強度 | candela, cd / 坎德拉 | Visible-light output per solid angle in a direction / 某方向每單位立體角的可見光輸出 |
| Luminous flux $\Phi_v$ / 光通量 | lumen, lm / 流明 | Total photometrically weighted light flow / 按人眼敏感度加權的總光流 |
| Illuminance $E_v$ / 照度 | lux, lx / 勒克斯 | Incident luminous flux per surface area / 表面每單位面積接收的光通量 |
| Solid angle $\Omega$ / 立體角 | steradian, sr / 球面度 | Angular extent of an area as viewed from a point / 從一點看某面積的角度範圍 |

**EN:** On a sphere of radius $r$, a patch of area $r^2$ subtends 1 sr. A whole sphere subtends $4\pi$ sr. For uniform intensity over a solid angle and uniform illumination over a receiving area,

**繁中：** 半徑為 $r$ 的球面上，面積 $r^2$ 對應 1 sr；整個球面為 $4\pi$ sr。若某立體角內強度均勻，且接收面上的照明均勻，則：

$$
\Phi_v=I_v\Omega,\qquad 1\ \mathrm{lm}=1\ \mathrm{cd}\cdot\mathrm{sr},\qquad
E_v=\frac{\Phi_v}{A},\qquad 1\ \mathrm{lx}=1\ \mathrm{lm/m^2}.
$$

**EN:** An isotropic 1 cd source emits $4\pi$ lm in total. Spreading 1000 lm uniformly over 1 m² produces 1000 lx; over 10 m² it produces 100 lx. A lumen describes light flow, not how much electric power the lamp consumes.

**繁中：** 各方向均為 1 cd 的理想均勻光源，總光通量為 $4\pi$ lm。把 1000 lm 均勻分布在 1 m²，照度為 1000 lx；分布在 10 m²，則為 100 lx。流明描述光流，不是燈具消耗多少電功率。

**EN:** Photometric quantities weight radiation by visual sensitivity. The candela definition uses monochromatic radiation at $540\times10^{12}$ Hz and radiant intensity $1/683$ W/sr for 1 cd. Hence equal radiant watts at different wavelengths need not give equal lumens. The lecture's approximate lighting examples include offices at 320-500 lx and a TV studio near 1000 lx; these are illustrative conditions, not universal standards.

**繁中：** 光度學的量會按視覺敏感度加權。坎德拉定義使用 $540\times10^{12}$ Hz 的單色光；當該方向輻射強度為 $1/683$ W/sr 時，發光強度為 1 cd。因此，不同波長即使輻射瓦數相同，流明也未必相同。講義舉出辦公室約 320 至 500 lx、電視錄影廠約 1000 lx，應視為示例條件，而非普遍標準。

### 5.13 Reflection and Refraction / 反射與折射

**Source / 來源：L05 pp. 30-32.**

**EN:** In specular reflection, the incident ray, reflected ray, and surface normal lie in the same plane, and incidence angle equals reflection angle, measured from the normal. Reflections can make a camera observe an apparent object position rather than a directly visible physical surface.

**繁中：** 鏡面反射中，入射線、反射線及表面法線位於同一平面，入射角等於反射角；角度從法線量起。反射可能令攝影機觀察到表觀物件位置，而不是直接可見的真實表面。

**EN:** Refraction changes a light ray's direction when it crosses media at an angle because propagation speed changes. Frequency stays constant at the boundary while wavelength changes. Lenses exploit refraction to form images; transparent objects and mirrors complicate scene interpretation.

**繁中：** 光以斜角穿越不同介質時，傳播速度改變可令光線方向改變，稱為折射。通過介面時頻率維持不變，波長則改變。鏡頭利用折射成像；透明物件和鏡面會增加場景解讀難度。

**Supplement / 補充:** Snell's law is $n_1\sin\theta_1=n_2\sin\theta_2$, where refractive index $n=c/v$. At normal incidence there need not be any change in direction despite a speed change. / 斯涅爾定律為 $n_1\sin\theta_1=n_2\sin\theta_2$，折射率 $n=c/v$。垂直入射時，即使速度改變，方向也未必改變。

### 5.14 Recording Light and Forming Images / 記錄光與影像形成

**Source / 來源：L05 pp. 33-36.**

**EN:** A camera obscura projects an image into a dark chamber but does not itself store it. Light-sensitive chemicals enabled persistent photographs; film developed this principle; electronic photosensors enable digital capture. The slide's history is a simplified progression, not a claim that chemical photography has entirely disappeared.

**繁中：** 暗箱把影像投射到暗室內，但本身不會儲存影像。感光化學物料讓照片能長久保存，菲林發展了這個原理，電子感光元件則支援數碼擷取。講義是簡化歷史脈絡，不代表化學攝影已完全消失。

**EN:** Image formation has two parts: **geometry** determines where a scene point is projected; **light physics** determines its recorded brightness from illumination, surface properties, and the imaging system. A camera maps a 3D world to a 2D image, so depth is generally not directly retained in one ordinary image.

**繁中：** 成像包含兩部分：**幾何** 決定場景點投影在哪裏；**光學物理** 則根據照明、表面性質及成像系統決定記錄亮度。攝影機把三維世界映射到二維影像，因此單張普通影像一般不直接保留深度。

**Supplement: projection / 補充：投影。** In an ideal pinhole model with a convenient virtual image plane in front of the camera, $x=fX/Z$, $y=fY/Z$. Here $(X,Y,Z)$ are coordinates relative to the camera, $f$ is focal distance, and all lengths use consistent units. A physical image plane behind the pinhole introduces image inversion/sign changes. Different depths can project to the same image point. / 在理想針孔模型中，若方便地使用相機前方虛擬影像平面，可寫 $x=fX/Z$、$y=fY/Z$。$(X,Y,Z)$ 為相對相機的座標，$f$ 為焦距，長度單位須一致。針孔後方的真實影像平面會出現倒像／符號改變。不同深度的點可能投影到同一影像位置。

### 5.15 CCD Versus CMOS / CCD 與 CMOS 比較

**Source / 來源：L05 pp. 37-38.**

| Aspect / 比較項目 | CCD | CMOS |
|---|---|---|
| Readout principle / 讀出原理 | Transfer accumulated charge through registers toward readout / 把累積電荷經移位暫存器傳送至讀出端 | Address pixels/rows and use integrated readout circuitry / 定址像素／行並使用整合讀出電路 |
| Charge-to-voltage conversion / 電荷轉電壓 | Usually at a shared output stage in the simplified model / 簡化模型中通常在共用輸出級 | Commonly within pixel-level circuitry / 常在像素級電路進行 |
| Integration / 整合 | More functions may be off-sensor in traditional designs / 傳統設計較多功能置於感測器外 | Readout/control functions can be highly integrated / 讀出及控制可高度整合 |
| Historical comparison / 歷史比較 | Slides emphasize sensitivity and low-light quality / 講義強調感光度及低光表現 | Slides emphasize integration and use in megapixel cameras / 講義強調整合及高像素攝影機應用 |

**Clarification / 釐清:** The slide's 'CCD is better in low light' is historical, not a universal present-day rule. Sensor size, pixel design, quantum efficiency, read noise, and processing matter. CMOS does not inherently require rolling shutter; global-shutter CMOS exists. Distinguish readout architecture from a particular exposure mode. / 投影片「CCD 低光較好」屬歷史概括，並非現今通則。感測器尺寸、像素設計、量子效率、讀出噪音及處理都重要。CMOS 亦不是必然採用捲動快門；全域快門 CMOS 確實存在。應區分讀出架構與特定曝光模式。

### 5.16 Resolution, Frame Rate, and Video / 解像度、影格率與影片

**Source / 來源：L05 pp. 39-40.**

**EN:** Nominal image resolution is often expressed as width × height pixels; megapixels count millions of pixels. A still camera captures individual images; a video camera captures an image sequence. FPS is frames per second and describes temporal sampling, not spatial detail. Video is commonly compressed and packaged into a file or stream.

**繁中：** 標稱影像解像度通常以「寬 × 高」像素表示；百萬像素計算的是像素數。照片相機擷取個別影像，影片攝影機擷取影像序列。FPS 指每秒影格數，描述時間取樣，不是空間細節。影片通常會被壓縮並封裝成檔案或資料流。

**EN:** Exposure duration and frame rate are related but different controls. A shorter exposure reduces motion accumulated within one frame but collects less light. More frames per second provides finer temporal sampling but increases raw data rate if resolution and bit depth stay fixed.

**繁中：** 曝光時間與影格率有關，但屬不同控制。較短曝光可減少單幀內累積的動態模糊，但收光較少。若解像度及位元深度不變，每秒更多影格提供較細時間取樣，也增加原始資料率。

**Clarification / 釐清:** The slide's 32 MP still-camera and 16 MP video-camera figures are historical examples, not current upper limits. Pixel count alone does not determine image quality, and one photosite is not necessarily a complete independently measured RGB triplet because colour-filter arrays may require reconstruction. / 投影片所指照片相機 32 MP、影片相機 16 MP 是歷史例子，不是現今上限。像素數不單獨決定畫質；一個感光點也未必獨立量度完整 RGB，因為彩色濾鏡陣列可能需要重建顏色。

### 5.17 Stereo Cameras and Depth / 立體攝影機與深度

**Source / 來源：L05 p. 41; connects to L04 pp. 12-15.**

**EN:** Two cameras observe a scene from separated viewpoints. The position difference of a matched point is its **disparity**. For an ideal rectified pair with parallel optical axes,

**繁中：** 兩部攝影機從分開的視點觀察場景。同一點在兩幅影像的位置差稱為 **視差**。對已校正、光軸平行的理想雙攝影機：

$$
Z=\frac{f_{\mathrm{px}}b_{\mathrm{cam}}}{d_{\mathrm{px}}}.
$$

**Supplement / 補充:** $Z$ is depth, $f_{\mathrm{px}}$ is focal length in pixels, $b_{\mathrm{cam}}$ is camera baseline in metres, and $d_{\mathrm{px}}$ is disparity in pixels; $Z$ is then in metres. The slide introduces stereo depth qualitatively; this formula is an explanatory extension. / $Z$ 為深度，$f_{\mathrm{px}}$ 為像素單位焦距，$b_{\mathrm{cam}}$ 為公尺單位的相機基線，$d_{\mathrm{px}}$ 為像素視差，因此 $Z$ 的單位為公尺。投影片只定性介紹立體深度，此公式屬解說延伸。

**EN:** Larger disparity implies smaller depth for fixed focal length and baseline. Correspondence is difficult for textureless areas, repetitive patterns, reflections, and occluded points. Calibration and synchronization are important, particularly for moving scenes.

**繁中：** 焦距及基線固定時，視差愈大，深度愈小。缺乏紋理、重複圖樣、反射及遮擋會令對應點搜尋困難。校準和同步都很重要，尤其在動態場景中。

### 5.18 Multiple Cameras, ToF, Folded Optics, and Gestures / 多鏡頭、ToF、摺疊光學與手勢

**Source / 來源：L05 pp. 42-46.**

**EN:** Multiple phone cameras can offer different fields of view, focal lengths, and sensing capabilities. They may support zoom transitions, depth estimation, or fusion, but simply having two cameras does not guarantee calibrated stereo depth. Different viewpoints and colour responses must be reconciled when combining their output.

**繁中：** 手機多鏡頭可提供不同視野、焦距及感測能力，並支援變焦切換、深度估計或融合，但「有兩個鏡頭」不保證已有校準立體深度。若要合併輸出，須處理不同視角及顏色響應。

**EN:** A **Time-of-Flight (ToF)** sensor actively emits light, often infrared, and estimates depth from its return. The slide specifically shows modulated IR and a measured phase shift, followed by a depth calculation, postprocessing, and application interpretation. It also shows return amplitude as a quality indicator. Passive stereo instead infers depth from image correspondences without requiring its own illumination.

**繁中：** **飛行時間（ToF）** 感測器會主動發光，通常是紅外線，再根據返回訊號估計深度。投影片特別顯示調制紅外線及量度相位差，之後進行深度計算、後處理及應用解讀，並以返回振幅作為品質指標。被動立體視覺則從影像對應推斷深度，不一定需要主動照明。

**Supplement / 補充:** For direct round-trip timing, $d=c\Delta t/2$; the factor 2 accounts for outward and return travel. Phase-based ToF uses modulation phase rather than directly timing each individual pulse and can have phase ambiguity. Strong ambient light, weak reflection, and multiple reflected paths can affect measurements. / 對直接來回計時，$d=c\Delta t/2$；除以 2 是因為光走了去程和回程。相位式 ToF 利用調制相位，不是直接逐一量度每個脈衝的時間，而且可能有相位歧義。強環境光、弱反射及多重反射路徑都可能影響量度。

**EN:** **Folded optics** bends the optical path with mirrors/prisms, allowing a longer effective path within a thin device. It is a packaging/optical design idea, not itself a depth-measurement method. The **Leap Motion** example illustrates optical hand/finger tracking for gesture input, linking sensor capture to intuitive experiential interaction.

**繁中：** **摺疊光學** 利用反射鏡／稜鏡折轉光路，在纖薄裝置內容納較長光學路徑。它是光學及封裝設計概念，本身不是深度量度方法。**Leap Motion** 例子展示以光學方式追蹤手和手指，作為手勢輸入，把感測擷取連結到直觀的體驗式互動。

### 5.19 Additive and Subtractive Colour / 加色法與減色法

**Source / 來源：L05 p. 47.**

**EN:** Additive reproduction combines emitted light. Ideal RGB primaries together can produce white; no emission gives black. Subtractive reproduction uses materials such as ink to absorb selected wavelengths from incident light. An unprinted white page reflects a broad range; adding inks removes portions of that reflected spectrum. Without illumination, ordinary printed paper does not emit the image.

**繁中：** 加色重現把發出的光組合起來。理想 RGB 原色合起來可產生白色；沒有發光便是黑色。減色重現利用油墨等材料吸收部分入射波長。未印刷白紙反射廣泛波段，加入油墨便減去部分反射頻譜。沒有照明時，普通印刷紙不會自行發出影像。

**EN:** RGB is associated with displays; CMYK with printing. K adds practical black reproduction and helps avoid relying on imperfect combined cyan/magenta/yellow inks. A colour possible on a display is not automatically reproducible by a printer because their gamuts differ.

**繁中：** RGB 常用於顯示器，CMYK 常用於印刷。K 提供實用黑色，避免完全依賴不完美的青、洋紅、黃油墨混合。顯示器可顯示的顏色不一定能由印刷機重現，因為兩者色域不同。

### 5.20 CRT, LCD, OLED, and AMOLED / CRT、LCD、OLED 與 AMOLED

**Source / 來源：L05 pp. 48-50.**

| Display / 顯示技術 | Main mechanism / 主要機制 | Consequence / 特性 |
|---|---|---|
| CRT / 陰極射線管 | Electron beams excite phosphors / 電子束激發螢光體 | Bulky tube; emitted light from phosphors / 機身較厚，螢光體發光 |
| LCD / 液晶顯示器 | Liquid crystals modulate a backlight through optical layers / 液晶經光學層調制背光 | Pixels generally modulate rather than emit light; leakage affects blacks / 像素通常調制而非自行發光；漏光影響黑位 |
| OLED / 有機發光二極體 | Organic elements emit light electrically / 有機元件通電發光 | Self-emissive pixels; dark pixels can be off / 像素自發光，暗像素可關閉 |
| AMOLED / 主動矩陣 OLED | Active-matrix circuitry controls OLED pixels / 主動矩陣電路控制 OLED 像素 | A driving architecture for OLED, not a separate colour space / 是 OLED 驅動架構，不是另一種色彩空間 |

**EN:** The lecture's LCD diagram connects polarization to controlled light transmission. OLED does not need an LCD-style backlight. Real comparisons also depend on brightness, efficiency, viewing angle, response, and lifetime; one mechanism does not make every product superior in all respects.

**繁中：** 講義的 LCD 圖把偏振與可控制透光連結起來。OLED 不需要 LCD 式背光。實際比較還要考慮亮度、效率、視角、反應及壽命；某種機制不代表所有相關產品在各方面都較優勝。

### 5.21 Autostereoscopic Displays and 'Holograms' / 裸眼立體顯示與「全息影像」

**Source / 來源：L05 pp. 51-52.**

**EN:** An autostereoscopic display sends different views to the left and right eyes without requiring glasses. The slide shows a switchable parallax barrier: in 3D mode, selected light paths are separated; in 2D mode, the barrier permits both eyes to see the ordinary view. Depth perception comes from binocular disparity.

**繁中：** 裸眼立體顯示器把不同視圖送往左右眼，不需要佩戴眼鏡。投影片顯示可切換視差屏障：3D 模式分開指定光路；2D 模式則讓兩眼看到普通畫面。深度感來自雙眼視差。

**EN:** The main limitation highlighted is the viewing **sweet spot**: outside suitable angles, views may mix, blur, or become incorrect. The next slide labels a Gatebox-style character display '3D hologram.' **Clarification:** a product's floating-image appearance or marketing name does not establish true holography, which reconstructs an optical wavefront. Do not treat all 3D-looking displays as the same technology.

**繁中：** 講義強調的主要限制是觀看 **最佳位置／甜蜜點**：偏離合適角度時，左右視圖可能混合、模糊或錯誤。下一頁把 Gatebox 類角色顯示稱為「3D hologram」。**釐清：** 懸浮影像外觀或產品名稱，不足以證明是真正重建光波前的全息技術。不要把所有看來立體的顯示當成同一技術。

### 5.22 The Eye, Cones, and Rods / 眼睛、視錐與視桿

**Source / 來源：L05 pp. 53-55.**

**EN:** The iris controls admitted light; the lens helps focus the image onto the retina; retinal receptors convert light into biological signals that ultimately reach the brain via the optic nerve. The retina is not a uniform camera array.

**繁中：** 虹膜控制入光量；晶狀體協助把影像聚焦在視網膜；視網膜感光細胞把光轉成生物訊號，再經視神經傳往大腦。視網膜並不是均勻分布的相機感測陣列。

| Property / 特性 | Cones / 視錐細胞 | Rods / 視桿細胞 |
|---|---|---|
| Dominant condition / 主要條件 | Photopic, bright-light vision / 明視覺，較明亮環境 | Scotopic, dim-light vision / 暗視覺，較昏暗環境 |
| Colour / 顏色 | Supports colour discrimination / 支援顏色分辨 | Does not provide comparable colour discrimination / 不提供同等顏色分辨能力 |
| Distribution / 分布 | Dense in the foveal region / 中央凹密度高 | More peripheral; not concentrated in the foveal centre / 較多在周邊，不集中於中央凹中心 |
| Detail and sensitivity / 細節及敏感度 | High central acuity / 中央視覺精細度高 | High dim-light sensitivity with spatial pooling / 透過空間匯聚提高暗光敏感度 |
| Approximate slide counts / 講義約數 | 6-7 million / 600 至 700 萬 | 75-150 million / 7,500 萬至 1.5 億 |

**Clarification / 釐清:** These counts are approximate, and retinal wiring is more complex than 'every cone has its own nerve.' The core contrast is fine central colour vision versus more sensitive, pooled dim-light vision. / 這些數量只是約數，視網膜連接亦比「每個視錐各接一條神經」複雜。核心對比是中央精細彩色視覺，以及較敏感、經匯聚的暗光視覺。

### 5.23 Resolving Power and Eye Movement / 分辨能力與眼球運動

**Source / 來源：L05 pp. 56-57.**

**EN:** The lecture approximates a 1.5 mm-wide foveal region as a square and uses a density of 150,000 cones/mm². Its estimate is $1.5\times1.5\times150{,}000=337{,}500$ receptors, about 337,000. This is a deliberately simplified local comparison with an imaging array, not the eye's total 'megapixel count.'

**繁中：** 講義把寬約 1.5 mm 的中央凹區近似成正方形，並使用每 mm² 150,000 個視錐的密度，估計為 $1.5\times1.5\times150{,}000=337{,}500$ 個，約 337,000。這是刻意簡化的局部感測陣列比較，不是整隻眼的「總百萬像素數」。

**EN:** Eyes actively move while exploring a scene. The brain integrates information across fixations rather than receiving one uniformly detailed static frame. This connects back to the perceptual cycle: the task and schema guide where we look next.

**繁中：** 眼睛探索場景時會主動移動。大腦整合多次注視取得的資訊，而不是接收一幅全畫面都同樣精細的靜態影像。這亦連結知覺循環：任務及基模引導下一次注視的位置。

### 5.24 Brightness Perception and Adaptation / 亮度知覺與適應

**Source / 來源：L05 pp. 58-59.**

**EN:** Physical intensity and perceived brightness are different. The lecture uses a logarithmic relationship as an explanatory model, but perception also depends on surrounding contrast and adaptation. Identical pixel intensities can appear different against different backgrounds.

**繁中：** 物理強度與感知亮度不同。講義以對數關係作解說模型，但知覺同時受周圍對比及適應狀態影響。完全相同的像素強度，在不同背景下可以看來不同。

**EN:** Brightness adaptation changes overall visual sensitivity. The lecture describes adaptation across an enormous intensity range, on the order of $10^{10}$, but explicitly notes that the eye cannot operate across that whole range simultaneously. Do not confuse the total adaptation range with the instantaneous dynamic range of one captured image.

**繁中：** 亮度適應會改變整體視覺敏感度。講義指出可適應的強度範圍極大，約達 $10^{10}$ 量級，但亦明確指出眼睛不能同時處理整個範圍。不要把整體適應範圍，混同單一影像瞬間可擷取的動態範圍。

### 5.25 Colour Spaces and True Tone / 色彩空間與 True Tone

**Source / 來源：L05 pp. 60-62.**

| Representation / 表示 | Use / 用途 | Key meaning / 重點 |
|---|---|---|
| RGB | Displays and image representation / 顯示及影像表示 | Additive red, green, blue components / 紅、綠、藍加色成分 |
| RGBA | Images needing transparency / 需要透明度的影像 | Alpha represents opacity/coverage, not a fourth visible primary / Alpha 表示不透明度／覆蓋率，不是第四種可見原色 |
| CMYK | Printing / 印刷 | Subtractive cyan, magenta, yellow, black / 青、洋紅、黃、黑減色 |
| YUV-family representations / YUV 類表示 | Video and image coding / 影片及影像編碼 | Separate a brightness-related component from chroma / 分離與亮度有關的成分及色度 |

**EN:** Separating luma and chroma allows coding choices that exploit human sensitivity differences, for example allocating less spatial detail to colour than to luma. **Clarification:** digital coding commonly uses YCbCr; 'YUV' is often used loosely. Luma is derived from encoded colour signals and is not identical to physical luminance. An RGB triplet alone also needs a specified colour space/profile for unambiguous colour meaning.

**繁中：** 分離亮度相關分量和色度後，編碼可利用人眼敏感度差異，例如給色度較少空間細節。**釐清：** 數碼編碼常用 YCbCr；「YUV」有時被寬鬆使用。Luma 從編碼顏色訊號推算，不完全等於物理亮度 luminance。單一 RGB 三元組亦需要指定色彩空間／描述檔，才能確定顏色意義。

**EN:** True Tone adjusts the display's white appearance using ambient-light information, aiming for a more natural match to the viewing environment. This is an example of **context affecting reproduction**: the same intended white may need different display settings under different illumination. For controlled colour-critical comparison, automatic adjustments need to be considered rather than assuming the display output is fixed.

**繁中：** True Tone 利用環境光資訊調整顯示器的白色外觀，使其更自然地配合觀看環境。這是 **情境影響重現** 的例子：不同照明下，要呈現自然白色，可能需要不同顯示設定。進行受控、顏色要求嚴格的比較時，須考慮自動調整，不能假設顯示輸出固定。

<a id="formulas"></a>
## 6. Formula Sheet and Worked Examples / 公式表及計算例題

### 6.1 Formula Sheet / 公式表

**EN:** 'Core' means directly used or defined in the slides; 'Extension' means a standard derivation or explanatory addition to a slide topic. This classification is about source grounding, not a guarantee of exam weighting. All logarithms in decibel formulas are base 10.

**繁中：** 「核心」指投影片直接使用或定義的關係；「延伸」指相關主題的標準推導或解說補充。這是來源分類，不代表考試分數比重。所有分貝公式中的對數均以 10 為底。

| Topic / 主題 | Formula / 公式 | Meaning and condition / 意義及條件 | Status / 分類 |
|---|---|---|---|
| Sinusoid / 正弦波 | $x(t)=A\sin(2\pi ft+\phi)$ | Amplitude, frequency, phase / 振幅、頻率、相位 | Extension of L04 pp. 28-30 / 延伸 |
| Period / 週期 | $T=1/f$ | $f$ in Hz, $T$ in s / 頻率以 Hz，週期以秒 | Extension / 延伸 |
| Sampling interval / 取樣間隔 | $T_s=1/f_s$ | Time between samples / 樣本之間的時間 | Core, L04 pp. 31-34 / 核心 |
| Quantization levels / 量化級數 | $L=2^b$ | $b$ bits per scalar sample / 每個純量樣本有 $b$ 位元 | Extension of L04 p. 33 / 延伸 |
| Uniform quantizer / 均勻量化 | $\Delta=R/2^b$ | Span $R$, equal-width bins; no overload / 範圍 $R$、等寬區間，未過載 | Extension / 延伸 |
| Nyquist / 奈奎斯特 | $f_N=f_s/2$, $f_s>2B$ | Band-limited baseband signal / 頻帶受限的基頻訊號 | Core, L04 pp. 33-34 / 核心 |
| Raw PCM audio / 原始 PCM 聲音 | $R_a=f_sbc$ | $c$ channels; result in bit/s / $c$ 個聲道；結果為 bit/s | Extension / 延伸 |
| Audio size / 聲音大小 | $S_a=f_sbct/8$ | Duration $t$ in s; result in bytes / 時間 $t$ 秒；結果為位元組 | Extension / 延伸 |
| Raw image / 原始影像 | $S_i=WHq/8$ | $q$ total bits/pixel / $q$ 是每像素總位元數 | Extension / 延伸 |
| Raw video / 原始影片 | $R_v=WHqF$ | $F$ FPS, no compression / $F$ 為 FPS，未壓縮 | Extension / 延伸 |
| SPL / 聲壓級 | $L_p=20\log_{10}(p/p_{\mathrm{ref}})$ | RMS pressure; reference stated / RMS 聲壓，須交代參考值 | Core, L04 p. 36 / 核心 |
| Amplitude gain / 振幅增益 | $G=20\log_{10}(A_2/A_1)$ | Same amplitude quantity and comparison conditions / 同一振幅物理量及比較條件 | Extension / 延伸 |
| Power gain / 功率增益 | $G=10\log_{10}(P_2/P_1)$ | Power ratio, not amplitude ratio / 功率比，非振幅比 | Extension / 延伸 |
| Wave speed / 波速 | $v=f\lambda$ | Use speed for the given medium / 使用指定介質中的速度 | Extension / 延伸 |
| Echo distance / 回聲距離 | $d=v\tau/2$ | Direct out-and-back path / 直接去程加回程 | Extension / 延伸 |
| Block duration / 區塊時長 | $t=N/f_s$ | $N$ samples in one channel / 一個聲道中的 $N$ 個樣本 | Extension of MP3 example / 延伸 |
| Luminous flux / 光通量 | $\Phi_v=I_v\Omega$ | Uniform intensity over $\Omega$ / 在 $\Omega$ 內強度均勻 | Core, L05 p. 28 / 核心 |
| Illuminance / 照度 | $E_v=\Phi_v/A$ | Uniform incident flux across area / 光通量均勻落在面積上 | Core, L05 p. 29 / 核心 |
| Stereo / 立體視覺 | $Z=f_{\mathrm{px}}b_{\mathrm{cam}}/d_{\mathrm{px}}$ | Calibrated rectified geometry / 已校準並校正的幾何 | Extension / 延伸 |
| Direct ToF / 直接 ToF | $d=c\Delta t/2$ | Round-trip light delay / 光的來回延遲 | Extension / 延伸 |
| Pinhole projection / 針孔投影 | $x=fX/Z$, $y=fY/Z$ | Virtual front image plane / 前方虛擬影像平面 | Extension / 延伸 |

**Units / 單位:** 1 byte = 8 bits; 1 kHz = 1000 Hz; 1 ms = $10^{-3}$ s; 1 μs = $10^{-6}$ s; 1 ns = $10^{-9}$ s; 1 nm = $10^{-9}$ m. Decimal MB = $10^6$ bytes; binary MiB = $2^{20}$ bytes. / 1 位元組 = 8 位元；其餘單位換算如上。十進制 MB 與二進制 MiB 不相同。

### W1. Sinusoid and Period / 正弦波與週期

**Question / 題目:** A signal is $x(t)=2\sin(2\pi\cdot1000t+\pi/2)$ V. Identify its parameters. / 訊號為 $x(t)=2\sin(2\pi\cdot1000t+\pi/2)$ V，辨認其參數。

**EN:** Amplitude is 2 V, frequency is 1000 Hz, phase is $\pi/2$ rad, and period is $1/1000=0.001$ s = 1 ms. At $t=0$, the value is 2 V. Do not mistake the coefficient $2\pi\cdot1000$ in radians/s for frequency in Hz.

**繁中：** 振幅為 2 V，頻率為 1000 Hz，相位為 $\pi/2$ 弧度，週期為 $1/1000=0.001$ 秒，即 1 ms。在 $t=0$ 時，訊號值為 2 V。不要把單位為弧度每秒的 $2\pi\cdot1000$，直接當成 Hz 頻率。

### W2. Sampling Rate and Bandwidth / 取樣率與頻寬

**Question / 題目:** Audio contains frequencies up to 7 kHz and is sampled at 16 kHz. Is the ideal sampling condition satisfied? / 聲音最高頻率為 7 kHz，以 16 kHz 取樣，是否符合理想取樣條件？

$$
2B=14\ \mathrm{kHz}<16\ \mathrm{kHz},\quad f_N=8\ \mathrm{kHz},\quad
T_s=\frac{1}{16000}=62.5\ \mu\mathrm{s}.
$$

**EN:** Yes, for the stated band-limited signal, with appropriate anti-alias filtering. The 7 kHz content lies below 8 kHz. A component exactly at 8 kHz would be at the boundary and could have phase ambiguity; 16-bit quantization would not change that fact.

**繁中：** 對題目所指的頻帶受限訊號，並有合適抗混疊濾波時，答案是符合。7 kHz 低於 8 kHz 上限。若成分恰好為 8 kHz，便位於邊界，可能有相位歧義；改用 16 位元量化亦不能改變這點。

### W3. Aliasing / 混疊，延伸例題

**Question / 題目:** A 7 kHz sinusoid reaches an 8 kHz sampler without anti-alias filtering. What lower-frequency alias can appear? / 7 kHz 正弦波未經抗混疊濾波便進入 8 kHz 取樣器，可能變成哪個較低頻率？

$$
f_{\mathrm{alias}}=|7000-8000|=1000\ \mathrm{Hz}.
$$

**EN:** It can appear as a 1 kHz component with the corresponding phase/sign relationship. In fact, $\sin(2\pi7000n/8000)=-\sin(2\pi1000n/8000)$. The sampled sequence alone cannot tell whether the original was the high-frequency signal or its low-frequency equivalent.

**繁中：** 它可能表現為具有相應相位／正負號關係的 1 kHz 成分。實際上，$\sin(2\pi7000n/8000)=-\sin(2\pi1000n/8000)$。只看樣本序列，不能判斷原本是高頻訊號還是其低頻等效訊號。

### W4. Quantization Resolution / 量化解像度，延伸例題

**Question / 題目:** Quantize a -1 V to +1 V range using 8 bits and equal-width bins. / 以 8 位元、等寬區間量化 -1 V 至 +1 V 的範圍。

$$
L=2^8=256,\qquad \Delta=\frac{2}{256}=0.0078125\ \mathrm{V}.
$$

**EN:** The step is 7.8125 mV. With rounding to the nearest midpoint and no overload, the maximum error magnitude is approximately 3.90625 mV. With the same range at 16 bits, the step is about 30.52 μV. These statements assume the bin convention defined above, not an endpoint-to-endpoint code convention.

**繁中：** 步距為 7.8125 mV。若四捨五入到最近區間中心且未過載，最大誤差絕對值約為 3.90625 mV。同一範圍改用 16 位元時，步距約 30.52 μV。以上採用前述區間定義，而不是把最小及最大代碼放在兩端點的定義。

### W5. Uncompressed Audio Size / 未壓縮聲音大小，延伸例題

**Question / 題目:** Find the raw size of 60 s of stereo audio at 44.1 kHz and 16 bits per channel sample. Ignore headers and compression. / 44.1 kHz、每聲道樣本 16 位元的立體聲，長 60 秒；忽略標頭及壓縮，求大小。

$$
R_a=44100\times16\times2=1{,}411{,}200\ \mathrm{bit/s}.
$$

$$
S_a=\frac{1{,}411{,}200\times60}{8}=10{,}584{,}000\ \mathrm{bytes}.
$$

**EN:** The bitrate is **1,411,200 bit/s**. Size is **10,584,000 bytes**, or 10.584 MB, approximately 10.09 MiB. Stereo doubles the mono rate. Do not multiply by two again after channel count has already been included, and do not confuse bits with bytes.

**繁中：** 位元率為 **1,411,200 bit/s**。大小為 **10,584,000 位元組**，即 10.584 MB，約 10.09 MiB。立體聲是單聲道資料率的兩倍。已計入聲道數後，不應再次乘二，也不要把位元與位元組混淆。

### W6. Raw Image and Video Rate / 原始影像及影片資料率，延伸例題

**Question / 題目:** A 1920 × 1080 RGB video uses 8 bits per colour channel at 30 FPS. Find raw frame size and bitrate. / 1920 × 1080 RGB 影片，每色彩通道 8 位元，30 FPS，求原始每幀大小及位元率。

$$
q=3\times8=24\ \mathrm{bits/pixel},\qquad
S_i=\frac{1920\times1080\times24}{8}=6{,}220{,}800\ \mathrm{bytes}.
$$

$$
R_v=1920\times1080\times24\times30=1{,}492{,}992{,}000\ \mathrm{bit/s}.
$$

**EN:** Each frame is **6,220,800 bytes**, or 6.2208 MB. Raw bitrate is **1,492,992,000 bit/s**, about 1.493 Gbit/s. One minute would require about 11.197 GB, ignoring audio, padding, metadata, and compression. Actual compressed video rates can be much lower and are not fixed by resolution alone.

**繁中：** 每幀為 **6,220,800 位元組**，即 6.2208 MB。原始位元率為 **1,492,992,000 bit/s**，約 1.493 Gbit/s。一分鐘約需 11.197 GB，未計聲音、填充、中繼資料及壓縮。實際壓縮影片可以小得多，而且不能單靠解像度決定位元率。

### W7. SPL and Pressure Ratios / 聲壓級與聲壓比

**Question / 題目:** What is the SPL of 0.2 Pa RMS in air? What if pressure doubles? / 空氣中 RMS 聲壓為 0.2 Pa，聲壓級是多少？若聲壓加倍又如何？

$$
L_p=20\log_{10}\left(\frac{0.2}{20\times10^{-6}}\right)
=20\log_{10}(10{,}000)=80\ \mathrm{dB\ SPL}.
$$

**EN:** Doubling to 0.4 Pa adds $20\log_{10}2\approx6.02$ dB, producing about 86.02 dB SPL. The result is not 160 dB. Multiplying pressure by ten would add 20 dB, giving 100 dB SPL.

**繁中：** 加倍至 0.4 Pa 會增加 $20\log_{10}2\approx6.02$ dB，得到約 86.02 dB SPL，而不是 160 dB。若聲壓增至十倍，便增加 20 dB，得到 100 dB SPL。

### W8. Adding Independent Sound Levels / 合併獨立聲級，延伸例題

**Question / 題目:** Two independent sources each contribute 60 dB SPL at a point. What is their combined level? / 某位置有兩個互不相關聲源，各貢獻 60 dB SPL，合起來是多少？

$$
L_{\mathrm{total}}=10\log_{10}\left(10^{60/10}+10^{60/10}\right)
\approx63.01\ \mathrm{dB\ SPL}.
$$

**EN:** For uncorrelated sources, mean-square pressures add. Do not add decibel numbers directly. Phase-coherent waves require a different calculation because constructive or destructive interference can occur.

**繁中：** 不相關聲源應把均方聲壓相加，不能把分貝數字直接相加。相位相關的波須用另一計法，因為可能發生相長或相消干涉。

### W9. Echo Distance / 回聲距離，延伸例題

**Question / 題目:** An echo from a wall returns after 0.12 s in air at 343 m/s. How far away is the wall in the simple out-and-back geometry? / 聲音在空氣中以 343 m/s 傳播，0.12 秒後收到牆的回聲；假設直接來回路徑，牆有多遠？

$$
d=\frac{343\times0.12}{2}=20.58\ \mathrm{m}.
$$

**EN:** The sound travelled 41.16 m in total, but the wall is 20.58 m away. Dividing by two is essential because the measured delay includes the return journey.

**繁中：** 聲音總共走了 41.16 m，但牆距離為 20.58 m。必須除以二，因為量到的時間包括回程。

### W10. MP3 Block Duration / MP3 區塊時長

**Question / 題目:** How long do 1152 samples represent at 44.1 kHz? / 在 44.1 kHz 下，1152 個樣本代表多長時間？

$$
t=\frac{1152}{44100}\ \mathrm{s}\approx0.026122\ \mathrm{s}=26.12\ \mathrm{ms}.
$$

**EN:** This explains the lecture's 'about 26 ms.' Do not divide by 44.1 unless you intend the result to be in milliseconds rather than seconds.

**繁中：** 這解釋了講義的「約 26 ms」。除以 44.1 時，所得單位是毫秒；若要用秒，應除以 44,100。

### W11. Light Frequency / 光頻率，延伸例題

**Question / 題目:** Find the frequency of 600 nm light in a vacuum. / 求真空中 600 nm 光的頻率。

$$
f=\frac{c}{\lambda}=\frac{3.00\times10^8}{600\times10^{-9}}
=5.00\times10^{14}\ \mathrm{Hz}.
$$

**EN:** Convert nanometres to metres first. In another medium, light's wavelength and speed change together while its frequency remains the same at the interface.

**繁中：** 先把納米轉成公尺。光進入另一介質時，波長和速度一同改變，但通過介面的頻率保持不變。

### W12. From Candela to Lumen to Lux / 從坎德拉到流明再到照度

**Question / 題目:** A source has uniform intensity 10 cd across 2 sr. All that flux falls uniformly on 4 m². Find flux and illuminance. / 光源在 2 sr 內均勻發光強度為 10 cd，全部光通量均勻照在 4 m²，求光通量及照度。

$$
\Phi_v=10\times2=20\ \mathrm{lm},\qquad E_v=\frac{20}{4}=5\ \mathrm{lx}.
$$

**EN:** Candela becomes lumen only after accounting for solid angle; lumen becomes lux only after accounting for receiving area. If the same 10 cd source were isotropic over a whole sphere, its total output would instead be $40\pi\approx125.66$ lm.

**繁中：** 坎德拉換成流明時要計入立體角；流明換成勒克斯時要計入接收面積。若同一 10 cd 光源向整個球面均勻發光，總光通量則為 $40\pi\approx125.66$ lm。

### W13. Stereo Depth / 立體視覺深度，延伸例題

**Question / 題目:** A rectified stereo pair has focal length 800 pixels, baseline 0.10 m, and disparity 40 pixels. / 已校正立體相機的焦距為 800 像素，基線為 0.10 m，視差為 40 像素。

$$
Z=\frac{800\times0.10}{40}=2.0\ \mathrm{m}.
$$

**EN:** At 20 pixels disparity, depth would be 4.0 m. Smaller disparity means a more distant point for fixed geometry. Focal length and disparity must use matching image units.

**繁中：** 若視差為 20 像素，深度會是 4.0 m。幾何固定時，視差較小代表點較遠。焦距和視差必須採用相容影像單位。

### W14. Direct ToF Distance / 直接 ToF 距離，延伸例題

**Question / 題目:** A directly timed light pulse returns after 20 ns. / 直接計時的光脈衝在 20 ns 後返回。

$$
d=\frac{(3.00\times10^8)(20\times10^{-9})}{2}=3.0\ \mathrm{m}.
$$

**EN:** The light travelled 6 m in total. This example uses direct timing; the phase-based sensor illustrated in L05 p. 44 estimates the delay through modulation phase instead.

**繁中：** 光總共走了 6 m。此例使用直接計時；L05 第 44 頁的相位式感測器，則從調制相位估計延遲。

### W15. Pinhole Projection / 針孔投影，延伸例題

**Question / 題目:** In the virtual-front-plane model, $f=10$ mm, $X=100$ mm, and $Z=1000$ mm. Find $x$. / 在前方虛擬影像平面模型中，$f=10$ mm、$X=100$ mm、$Z=1000$ mm，求 $x$。

$$
x=\frac{10\times100}{1000}=1\ \mathrm{mm}.
$$

**EN:** If depth doubles while $X$ remains fixed, $x$ becomes 0.5 mm. Perspective makes the same transverse size project smaller at a greater distance. State the image-plane convention if discussing the sign.

**繁中：** 若深度加倍而 $X$ 不變，$x$ 會變成 0.5 mm。透視令相同橫向大小在較遠距離投影得較小。若討論正負號，應交代影像平面慣例。

<a id="practice"></a>
## 7. Practice Questions with Model Answers / 練習題及參考答案

**EN:** These are original revision questions based on the supplied material, not past-paper questions or predictions. Try answering without looking, then compare the mechanisms and distinctions, not just keywords. Numerical practice is in W1-W15.

**繁中：** 以下為根據講義自編的溫習題，不是歷屆試題或試題預測。先遮住答案作答，再比較原理及概念區分，不要只核對關鍵詞。計算練習見 W1 至 W15。

### Q1. What Makes a System Meaningfully Multimedia? / 怎樣才算有意義的多媒體系統？

**Question / 題目:** Why is combining video, sound, and position more useful than treating each stream independently? / 為何整合影片、聲音及位置，比各自獨立處理更有用？

**EN:** Each stream observes only part of an event. Video can show a collision, sound can provide its timing or acoustic evidence, and position can locate it. Synchronization and context allow these partial observations to be correlated into a coherent event interpretation. The aim is relevant information, not merely a larger number of media files. Incorrect correlations can also create false conclusions.

**繁中：** 每條資料流只觀察事件的一部分。影片可顯示碰撞，聲音可提供時間或聲學證據，位置則可定位。同步及情境讓這些部分觀測組成一致事件解讀。目標是相關資訊，不是增加媒體檔案數量；錯誤關聯亦可能導致錯誤結論。

### Q2. Time, Space, and Addressees / 時間、空間及受眾

**Question / 題目:** Compare a live telephone call, a printed book, and an uploaded recorded lecture. / 比較即時電話、印刷書籍和上載的錄製課堂。

**EN:** A live telephone call crosses space but does not intrinsically preserve the exchange across time. A printed book preserves information, is portable, and supports many readers through replication. An uploaded lecture combines replay across time, network delivery across space, and scalable distribution to many learners. Each technology removes different communication constraints.

**繁中：** 即時電話跨越空間，但本身不會把對話保存至其他時間。印刷書籍保存資訊、可攜帶，並透過複製供多人閱讀。上載課堂結合跨時間重播、跨空間網絡傳送，以及向大量學生分發。不同技術解除不同通訊限制。

### Q3. Experiential Environment / 體驗式環境

**Question / 題目:** Give three properties of an experiential environment and explain why a spreadsheet is an example. / 列出體驗式環境三項特性，並解釋試算表為何是例子。

**EN:** The properties are direct/intuitive interaction, shared query and presentation spaces, and attention to user state/context. A spreadsheet displays values and relationships; editing a displayed cell acts as a query/manipulation, and the recalculated sheet presents the result in the same space. This supports exploration without translating every intention into a separate command language.

**繁中：** 三項特性是直接直觀、查詢與呈現使用同一空間，以及考慮使用者狀態／情境。試算表顯示數值和關係；修改已顯示的儲存格就是查詢／操作，重新計算後仍在相同空間呈現結果，支援探索，而不必把每項意圖轉成另一套指令語言。

### Q4. Formal Streams and Features / 資料流與特徵模型

**Question / 題目:** Explain $S_i$, $M_i$, and $F_{ij}$. Must the number of streams equal the number of media types? / 解釋 $S_i$、$M_i$ 及 $F_{ij}$。資料流數量是否必須等於媒體類型數？

**EN:** $S_i$ is the $i$th sensor stream; $M_i$ describes that stream; $F_{ij}$ is its $j$th extracted feature stream. Several streams can have the same type, so $n$ streams need not mean $n$ media types. Integration uses correlated features and metadata to produce insights, and user feedback can enter as another stream. Temporal alignment is necessary when reasoning about the same event.

**繁中：** $S_i$ 是第 $i$ 條感測資料流，$M_i$ 描述該資料流，$F_{ij}$ 是從中抽取的第 $j$ 條特徵流。多條資料流可以屬同一類型，因此 $n$ 條資料流不等於必有 $n$ 種媒體。整合利用相關特徵及中繼資料產生洞見，使用者回饋亦可成為另一條流。若分析同一事件，便須對齊時間。

### Q5. Perception and the Perceptual Cycle / 知覺與知覺循環

**Question / 題目:** Why does prior knowledge help interpret an ambiguous image? How is perception active? / 為何先驗知識有助理解含糊影像？知覺如何具主動性？

**EN:** The same signal can support several interpretations. Prior knowledge and context guide which hypotheses to test. In the perceptual cycle, a schema directs exploration, exploration samples the environment, and information updates the schema. An observer may move, zoom, or inspect a label to resolve ambiguity rather than passively accepting the first image.

**繁中：** 相同訊號可以支持多種解讀。先驗知識和情境引導要檢驗哪些假設。在知覺循環中，基模引導探索，探索從環境取樣，資訊再更新基模。觀察者可以移動、放大或查看標籤來消除歧義，而不是被動接受第一眼影像。

### Q6. Semantic Gap, Context, and Metadata / 語意鴻溝、情境與中繼資料

**Question / 題目:** A user asks for 'a suspicious person entering the building.' Why are pixels and timestamps insufficient by themselves? / 使用者要求找出「進入大廈的可疑人物」，為何單靠像素和時間戳記不足？

**EN:** Pixels are low-level content and timestamps provide limited context. 'Person,' 'entering,' and 'suspicious' involve object identity, temporal relationships, zone definitions, and application-specific judgement. Detection, tracking, contextual rules, and possibly other evidence are needed. Metadata helps interpretation but does not automatically prove intent; 'suspicious' must be operationally defined and uncertainty retained.

**繁中：** 像素是低階內容，時間戳記只提供部分情境。「人物」、「進入」及「可疑」涉及物件身份、時間關係、區域定義及應用特定判斷，須配合偵測、追蹤、情境規則及可能的其他證據。中繼資料有幫助，但不會自動證明意圖；「可疑」須有操作性定義，並保留不確定性。

### Q7. Shazam Versus Speech Recognition / Shazam 與語音辨識

**Question / 題目:** Both use time-frequency analysis. Why are their outputs different? / 兩者都使用時頻分析，為何輸出不同？

**EN:** Shazam matches recording fingerprints to a catalogue and returns a recording identity. Speech recognition interprets acoustic evidence using phonetic/language models to return words. A Shazam query can be a noisy mid-song excerpt because relative peak structure still matches the stored recording. Recognizing previously unseen sentences requires linguistic generalization, not merely identifying one stored waveform.

**繁中：** Shazam 把錄音指紋與目錄比對，輸出錄音身份；語音辨識則利用語音／語言模型解讀聲學證據，輸出詞語。Shazam 可使用有噪音的歌曲中段，因為相對峰值結構仍可能吻合。辨認未出現過的句子需要語言泛化，不是只識別某條已儲存波形。

### Q8. LPC Versus MPEG Audio / LPC 與 MPEG 聲音

**Question / 題目:** Compare what each method preserves and exploits. / 比較兩種方法保留及利用甚麼。

**EN:** LPC uses a speech-production model and communicates parameters such as filter coefficients, pitch, energy, and voicing. MPEG perceptual coding analyses audibility and allocates bits according to masking and coding redundancy. Frequency masking concerns nearby spectral components; temporal masking concerns nearby times. Both can be lossy, but their assumptions and suitable signal classes differ.

**繁中：** LPC 使用發聲模型，傳送濾波器係數、基音、能量及有聲狀態等參數。MPEG 知覺編碼分析可聽程度，按遮蔽及編碼冗餘分配位元。頻率遮蔽涉及頻譜鄰近成分，時間遮蔽涉及時間鄰近聲音。兩者都可以有損，但假設及適用訊號種類不同。

### Q9. ANC, Microphone Noise Reduction, and EQ / ANC、麥克風降噪與等化

**Question / 題目:** Are these three operations interchangeable? / 這三種操作是否可以互相替代？

**EN:** No. ANC produces acoustic anti-noise at a target location. Microphone noise reduction estimates unwanted components in a captured electrical/digital signal. EQ changes gain by frequency band, mainly altering tonal balance. EQ cannot generally distinguish desired speech and noise occupying the same bands, and microphone denoising does not itself quiet the air around the listener.

**繁中：** 不可以。ANC 在目標位置產生聲學反噪音；麥克風降噪估計並減少已擷取電氣／數碼訊號中的非目標成分；等化則按頻帶調整增益，主要改變音色平衡。EQ 一般不能區分同頻帶的語音和噪音，而錄音降噪本身也不會令聽者周圍空氣更安靜。

### Q10. Stabilization and Deblurring / 防震與去模糊

**Question / 題目:** Why may a stabilized video still look blurry? Outline the lecture's deblurring workflow. / 為何防震後影片仍可能模糊？概述講義的去模糊流程。

**EN:** Stabilization corrects frame-to-frame movement; blur forms during each exposure. The workflow extracts a foreground mask, segments/detects corners, tracks features, estimates motion, computes blur kernels, and deblurs. A single global motion model may fail when objects or different body parts move differently. Noise and lost detail limit restoration.

**繁中：** 防震修正幀與幀之間的移動，模糊卻在每次曝光期間形成。流程為抽取前景遮罩、分割／偵測角點、追蹤特徵、估計動作、計算模糊核，以及去模糊。物件或身體不同部分移動不一致時，單一全域模型可能失效。噪點及已流失細節亦限制還原。

### Q11. Face Detection and 3D Reconstruction / 人臉偵測與三維重建

**Question / 題目:** Compare Haar/AdaBoost detection with AAM-based reconstruction. / 比較 Haar／AdaBoost 偵測與 AAM 式重建。

**EN:** Haar/AdaBoost detection decides whether image regions contain faces, scanning positions and scales. AAM-based reconstruction locates detailed facial features, selects a suitable view model, fits 3D shape, estimates pose, rebuilds texture, and renders a normalized face. Reconstruction can reduce pose mismatch for recognition, but hidden surfaces require assumptions and a rendered frontal image is not a new direct observation.

**繁中：** Haar／AdaBoost 偵測透過掃描位置及尺度，判斷影像區域是否含人臉。AAM 式重建則定位細緻臉部特徵、選擇視角模型、擬合三維形狀、估計姿態、重建紋理，再渲染標準化人臉。重建可減少辨識中的姿態不匹配，但隱藏表面要靠假設，渲染正面圖不是新的直接觀測。

### Q12. Gait and Video Retrieval / 步態與影片檢索

**Question / 題目:** Why can a codeword representation help search, yet still lose meaning? / 碼字表示為何有助搜尋，卻仍可能流失意義？

**EN:** Codewords turn complex silhouettes into compact comparable features. Accumulated codeword counts can support indexing or matching, while colour/edge descriptors provide other searchable cues. However, compact descriptors may discard exact arrangement or temporal order. A similar feature vector does not automatically mean the same person or event; tracking, context, and further verification remain useful.

**繁中：** 碼字把複雜輪廓轉成緊湊、可比較的特徵。累積碼字數量可支援索引或比對，顏色／邊緣描述亦提供其他搜尋線索。但緊湊描述可能刪去精確排列或時間順序。特徵向量相似，不自動代表同一人或事件；追蹤、情境及進一步驗證仍有價值。

### Q13. Sensors Versus Actuators / 感測器與致動器

**Question / 題目:** Classify a microphone, accelerometer, GPS receiver, vibration motor, and speaker. / 分類麥克風、加速度計、GPS 接收器、震動馬達及揚聲器。

**EN:** The microphone senses acoustic pressure; the accelerometer senses acceleration-related input; GPS estimates location/timing from received signals. These provide input. The vibration motor and speaker produce physical output and are actuators/transducers used for reproduction. A multimedia interaction can include both sensing a touch and producing tactile feedback, but they are separate functions.

**繁中：** 麥克風感測聲壓，加速度計感測與加速度相關的輸入，GPS 從接收訊號估計位置／時間，三者提供輸入。震動馬達及揚聲器產生物理輸出，屬用於重現的致動器／轉換器。多媒體互動可以同時包含觸碰感測及觸覺回饋，但它們是不同功能。

### Q14. Sensor Errors / 感測誤差

**Question / 題目:** Distinguish bias, nonlinearity, hysteresis, drift, and saturation. / 區分偏差、非線性、遲滯、漂移和飽和。

**EN:** Bias is a baseline offset; nonlinearity is a non-proportional input/output relationship; hysteresis depends on prior input history; drift changes with time; saturation exceeds the usable range. Calibration may correct repeatable bias and modelled nonlinearity. It cannot reconstruct arbitrary variations flattened by saturation. Hysteresis is not simply another name for drift.

**繁中：** 偏差是基線偏移；非線性是輸入輸出不成比例；遲滯取決於過去輸入歷程；漂移隨時間改變；飽和則超出可用量程。校準可能修正可重複偏差及可建模非線性，但不能重建飽和後被壓平的任意變化。遲滯不是漂移的另一名稱。

### Q15. Sampling Versus Quantization / 取樣與量化

**Question / 題目:** Can 24-bit recording compensate for too low a sampling rate? Why is the Nyquist boundary dangerous? / 24 位元能否補償取樣率太低？為何奈奎斯特邊界有風險？

**EN:** No. Bit depth controls amplitude resolution; sampling rate controls temporal information and the representable frequency band. At $f_s=2B$, a sinusoid at $B$ can have all-zero samples for a particular phase. Use a suitable band limit, anti-alias filtering, and a rate above twice the desired upper frequency. Already aliased data generally cannot be uniquely repaired afterward.

**繁中：** 不能。位元深度控制振幅解像度，取樣率控制時間資訊及可表示頻帶。在 $f_s=2B$ 時，頻率為 $B$ 的正弦波可因特定相位而全部取樣為零。應限制頻帶、使用抗混疊濾波，並使取樣率高於最高頻率兩倍。混疊後一般不能唯一地修復原訊號。

### Q16. Recording Gain / 錄音增益

**Question / 題目:** Why are both extremely low gain and extremely high gain undesirable? / 為何增益太低及太高都不好？

**EN:** Low gain leaves weak signals near noise and coarse quantizer steps, wasting available range. High gain clips peaks or overloads the analogue front end. Proper gain staging balances resolution and headroom. Digital amplification of a weak recording raises noise too, while lowering a clipped recording does not recover its original peaks.

**繁中：** 增益太低令微弱訊號接近噪音及粗量化步距，浪費可用量程；太高則削波或令類比前端過載。合適增益須平衡解像度及餘量。事後數碼放大弱錄音也會放大噪音，而調低已削波錄音不能恢復原本峰值。

### Q17. Acoustics and dB(A) / 聲學與 dB(A)

**Question / 題目:** Explain why a machine can struggle in a room that sounds acceptable to a person, and why SPL is not the same as dBA. / 為何人在房間覺得聲音尚可，機器卻可能辨識困難？SPL 為何不等於 dBA？

**EN:** Reflections and reverberation smear evidence even when a listener does not hear separate echoes. Interference changes measured amplitude. SPL expresses an RMS pressure ratio to a reference; dBA additionally applies frequency weighting reflecting aspects of hearing sensitivity. The conversion depends on spectrum, so there is no universal broadband dB-to-dBA offset.

**繁中：** 反射和混響即使未被人聽成獨立回聲，也會模糊辨識線索；干涉亦會改變量度振幅。SPL 是相對參考值的 RMS 聲壓比；dBA 另外加入反映部分聽覺敏感度的頻率加權。轉換取決於頻譜，因此沒有通用寬頻 dB 至 dBA 的固定差值。

### Q18. Virtual Surround / 虛擬環繞聲

**Question / 題目:** How can a system suggest a sound direction without a speaker at that location? / 沒有在某方向放揚聲器，系統如何仍能營造該方向的聲音？

**EN:** It manipulates interaural timing, level, and direction-dependent filtering so the ear signals resemble those from a real source. These cues include ITD, ILD, and HRTF effects. Listener anatomy, room reflections, head movement, and position affect accuracy. Virtual surround therefore reproduces localization cues rather than literally placing a physical source at every perceived direction.

**繁中：** 系統調整雙耳時間、聲級和方向相關濾波，使耳朵接收的訊號近似真實聲源，包括 ITD、ILD 及 HRTF 效應。身體結構、房間反射、頭部移動及位置都影響準確度。因此，虛擬環繞聲重現的是定位線索，而非真的在每個感知方向放置聲源。

### Q19. Pitch, Formants, Timbre, and ADSR / 音高、共振峰、音色與 ADSR

**Question / 題目:** Two instruments play 440 Hz, but sound different. Two vowels have the same pitch, but are distinguishable. Explain. / 兩種樂器同奏 440 Hz 卻不同音色；兩個元音音高相同卻能區分，試解釋。

**EN:** Equal fundamental frequency gives similar pitch, not identical spectrum or envelope. Instruments differ in harmonic/partial strengths and attack, decay, sustain, release behaviour. Vowels can share a fundamental while different vocal-tract shapes produce different formant patterns. Harmonics follow integer multiples of the fundamental; formants are resonance regions and should not be confused with individual harmonics.

**繁中：** 相同基頻帶來相近音高，不代表頻譜或包絡相同。樂器的諧波／分音強度，以及起音、衰減、持續和釋音行為不同。元音可有相同基頻，但不同聲道形狀產生不同共振峰。諧波位於基頻整數倍；共振峰是共振區域，不應與個別諧波混淆。

### Q20. Light Units / 光的單位

**Question / 題目:** Why can the same lamp produce different lux values without changing its total lumens? / 同一盞燈為何總流明不變，照度卻可以不同？

**EN:** Lumens describe luminous flux; lux describes flux per receiving area. Spreading the same incident flux over a larger area lowers average illuminance. Candela describes intensity in a direction per solid angle, so beam distribution also matters. Electrical wattage is energy consumption, not a direct substitute for any of these photometric quantities.

**繁中：** 流明描述光通量，勒克斯描述接收面每單位面積的光通量。相同入射光通量分布在更大面積，平均照度便下降。坎德拉描述某方向每單位立體角的發光強度，因此光束分布亦重要。電瓦數是能量消耗，不能直接代替這些光度學量。

### Q21. Stereo, ToF, and Folded Optics / 立體視覺、ToF 與摺疊光學

**Question / 題目:** Which of these measures depth, and how? / 哪些用來量度深度？原理是甚麼？

**EN:** Stereo estimates depth from corresponding image points and disparity, requiring calibration and usable texture. ToF actively illuminates and estimates return delay or modulation phase, with different failure modes such as multipath or weak returns. Folded optics redirects the optical path to fit a device; it does not alone measure depth. Two phone cameras may have different roles rather than form a stereo pair.

**繁中：** 立體視覺利用對應點和視差估計深度，需要校準及可用紋理。ToF 主動照明，估計返回延遲或調制相位，可能受多重路徑或弱返回訊號影響。摺疊光學只把光路折轉以配合機身，本身不會量深。手機兩鏡頭也可能各有用途，而不是立體相機組。

### Q22. CCD/CMOS, Resolution, and FPS / CCD／CMOS、解像度與 FPS

**Question / 題目:** Does more megapixels or a CCD label guarantee better video? / 較多百萬像素或使用 CCD，是否保證影片更好？

**EN:** No. Pixel count concerns nominal spatial sampling; FPS concerns temporal sampling. Noise, sensor size, lens quality, exposure, processing, and compression also affect quality. CCD/CMOS identify readout architectures, not a universal quality ranking. A high-resolution blurred/noisy frame may reveal less useful detail than a lower-resolution well-exposed frame.

**繁中：** 不保證。像素數關乎標稱空間取樣，FPS 關乎時間取樣；噪點、感測器尺寸、鏡頭、曝光、處理和壓縮亦影響畫質。CCD／CMOS 表示讀出架構，不是普遍品質排名。高像素但模糊／噪點多的畫面，可能比低像素而曝光良好的畫面更少有用細節。

### Q23. Displays and Colour Representations / 顯示器與色彩表示

**Question / 題目:** Compare LCD/OLED, RGB/CMYK, and luma/chroma separation. / 比較 LCD／OLED、RGB／CMYK，以及亮度／色度分離。

**EN:** LCD modulates a backlight; OLED pixels emit light. RGB is additive light mixing, while CMYK printing subtracts wavelengths from incident light. Luma/chroma representations separate components useful for perceptual coding; digital video commonly uses YCbCr even when casually called YUV. Alpha is transparency/coverage information, not an extra visible primary. None of these terms is interchangeable with image resolution.

**繁中：** LCD 調制背光，OLED 像素自發光。RGB 是光的加色混合，CMYK 印刷則從入射光減去部分波長。亮度／色度表示分離有助知覺編碼的成分；數碼影片常用 YCbCr，即使口語有時稱 YUV。Alpha 是透明度／覆蓋率，不是額外可見原色。這些術語都不等於解像度。

### Q24. Vision and Context-Aware Reproduction / 視覺與情境化重現

**Question / 題目:** Why is the eye not just a fixed camera, and how does True Tone use context? / 為何眼睛不只是固定相機？True Tone 如何利用情境？

**EN:** Cones and rods differ in distribution, acuity, colour sensitivity, and operating light levels. Eye movements actively sample a scene; brightness adaptation changes sensitivity, and perceived brightness depends on surrounding context. True Tone uses ambient-light information to adjust display white appearance. This links perception, sensor input, and output adaptation rather than assuming identical pixel values always produce identical experience.

**繁中：** 視錐和視桿在分布、精細度、色彩敏感度及適用光度方面不同。眼球移動主動探索場景；亮度適應改變敏感度，感知亮度亦受周圍情境影響。True Tone 利用環境光調整顯示白色外觀，把知覺、感測輸入及輸出適應連起來，而不是假設相同像素值永遠產生相同體驗。

### Q25. Integrated Design Question / 綜合系統設計題

**Question / 題目:** Design an elderly-care fall-alert system using the course concepts. Explain one benefit of fusion and at least three limitations. / 利用本課概念設計長者跌倒警報系統，解釋融合的一項好處及至少三項限制。

**EN:** Capture camera frames, microphone audio, wearable acceleration, and optional heart-rate information. Digitize with suitable sampling and dynamic range; attach timestamps and sensor metadata. Detect posture/motion, unusual acceleration, and acoustic cues, then align them in a common event window. Fuse evidence with context such as room location and the person's normal activity. Present a concise alert with supporting evidence and let a user confirm or dismiss it; feedback can improve later interpretation. A dropped wearable alone is less convincing than synchronized body-fall and distress evidence. Limitations include occlusion, noise/reverberation, device non-wear, clock mismatch, sensor bias, and false alarms from ordinary activity. **Supplementary design consideration:** protect privacy and restrict access to sensitive recordings.

**繁中：** 擷取攝影機畫面、麥克風聲音、穿戴裝置加速度，以及可選心率資訊。以合適取樣率和動態範圍數碼化，附上時間戳記及感測器中繼資料。偵測姿態／動作、異常加速度和聲學線索，再對齊至共同事件時窗。結合房間位置及個人正常活動等情境融合證據。輸出簡潔警報及支持證據，讓使用者確認或取消；回饋可改善之後詮釋。單是穿戴裝置掉落，說服力不及身體跌倒及求救同步出現。限制包括遮擋、噪音／混響、沒有佩戴裝置、時鐘不同步、感測偏差，以及日常活動造成的誤報。**補充設計考慮：** 保護私隱並限制敏感錄影存取。

<a id="quick-review"></a>
## 8. Final Revision Checklist / 最後溫習清單

### 8.1 Explain Without Looking / 不看筆記也能解釋

- [ ] Multimedia is integration of partial correlated information, not merely several file types. / 多媒體重點是整合部分且相關的資訊，不只是多種檔案類型。
- [ ] Time, space, and addressee constraints; compare writing, telephone, broadcasting, and recordings. / 時間、空間及受眾限制；比較文字、電話、廣播和錄製媒體。
- [ ] Experiential environments: direct/intuitive, shared query/presentation space, state/context. / 體驗式環境：直接直觀、查詢呈現同空間、狀態與情境。
- [ ] $S_i$, $M_i$, $F_{ij}$, multiple streams versus multiple types, synchronization, feedback. / 資料流、中繼資料、特徵流、流數與類型數、同步及回饋。
- [ ] Experience/information/knowledge; dictionaries/percepts; objects/events. / 經驗、資訊、知識；字典與知覺單元；物件與事件。
- [ ] Perception plus prior knowledge; schema -> exploration -> environment -> updated schema. / 知覺與先驗知識；基模 -> 探索 -> 環境 -> 更新基模。
- [ ] Semantic gap; content versus context versus metadata. / 語意鴻溝；內容、情境與中繼資料的分別。
- [ ] Shazam fingerprinting versus speech recognition; LPC versus psychoacoustic compression. / Shazam 指紋與語音辨識；LPC 與心理聲學壓縮。
- [ ] Frequency masking versus temporal masking; lossy coding versus coding redundancy. / 頻率遮蔽與時間遮蔽；有損編碼與編碼冗餘。
- [ ] Enhancement, stabilization, deblurring; explain the foreground-to-motion-to-kernel pipeline. / 增強、防震、去模糊；解釋前景、動作及模糊核流程。
- [ ] Detection versus tracking versus recognition; AAM, 3D mesh, pose, texture, frontalization. / 偵測、追蹤與辨識；AAM、三維網格、姿態、紋理及正面化。
- [ ] Gait codewords, per-object video indexing, colour/edge features, DIKW. / 步態碼字、逐物件影片索引、顏色／邊緣特徵及 DIKW。
- [ ] Sensor versus actuator; bias, nonlinearity, hysteresis, drift, dynamic range. / 感測器與致動器；偏差、非線性、遲滯、漂移及動態範圍。
- [ ] Sampling versus quantization; Nyquist rate versus Nyquist frequency; aliasing and filtering. / 取樣與量化；奈奎斯特取樣率與奈奎斯特頻率；混疊及濾波。
- [ ] SPL versus dBA; echo, reverberation, interference; microphone patterns and clipping. / SPL 與 dBA；回聲、混響、干涉；麥克風指向模式及削波。
- [ ] Crossovers, enclosures, 5.1 notation, ILD, ITD, HRTF. / 分頻器、音箱、5.1 表示法、ILD、ITD 及 HRTF。
- [ ] Pitch versus formants versus timbre; harmonics/overtones; ADSR; voiced/unvoiced speech. / 音高、共振峰與音色；諧波／泛音；ADSR；有聲／無聲語音。
- [ ] Coherence versus polarization; reflection versus refraction; cd/lm/lx and steradians. / 相干與偏振；反射與折射；cd、lm、lx 及球面度。
- [ ] CCD/CMOS, resolution/FPS/exposure, stereo/ToF, multiple cameras, folded optics. / CCD／CMOS、解像度／FPS／曝光、立體視覺／ToF、多鏡頭及摺疊光學。
- [ ] RGB/CMYK/YUV-family/RGBA; CRT/LCD/OLED; autostereoscopic sweet spots. / RGB／CMYK／YUV 類／RGBA；CRT／LCD／OLED；裸眼立體最佳觀看位置。
- [ ] Cones/rods, foveal acuity, eye movement, brightness adaptation, True Tone. / 視錐／視桿、中央凹精細視覺、眼球運動、亮度適應及 True Tone。

### 8.2 Calculate and State Assumptions / 能計算並交代假設

- [ ] Convert Hz/kHz, s/ms/μs/ns, m/nm, bits/bytes, MB/MiB correctly. / 正確換算頻率、時間、長度、位元／位元組及 MB／MiB。
- [ ] Compute sample interval, Nyquist frequency, quantization levels, and a simple alias. / 計算取樣間隔、奈奎斯特頻率、量化級數及簡單混疊。
- [ ] Compute uncompressed audio/image/video size, including channels and bits per pixel. / 計算未壓縮聲音／影像／影片大小，正確計入聲道和每像素位元數。
- [ ] Compute SPL from RMS pressure without confusing amplitude and power logarithms. / 由 RMS 聲壓計算 SPL，不混淆振幅與功率的對數公式。
- [ ] Convert intensity -> flux -> illuminance with the stated angle and area. / 按指定立體角及面積，把發光強度轉成光通量及照度。
- [ ] Explain round-trip factors in echo/ToF and inverse disparity in stereo, as extensions. / 作為延伸，解釋回聲／ToF 的來回因子，以及立體深度與視差的反比。

### 8.3 Short Answers Worth Rehearsing / 值得背熟的短答句

| Prompt / 提示 | Compact answer / 精簡答案 |
|---|---|
| Why multimedia? / 為何多媒體？ | Each medium supplies partial evidence; contextual correlation can produce a holistic account. / 每種媒介提供部分證據，情境化關聯有助建立整體理解。 |
| Why metadata? / 為何需要中繼資料？ | It describes acquisition and supports alignment and interpretation, but may be incomplete or wrong. / 描述擷取條件，支援對齊及詮釋，但可能不完整或錯誤。 |
| Why semantic gap? / 為何有語意鴻溝？ | Machine features and human objects/events are different abstraction levels. / 機器特徵與人類物件／事件位於不同抽象層次。 |
| Why strict Nyquist margin? / 為何需奈奎斯特餘量？ | The boundary can lose phase evidence; real filters also need transition space. / 邊界可流失相位證據，實際濾波亦需要過渡空間。 |
| Why not just increase bits? / 為何不只增加位元？ | Bits refine amplitude; they do not restore missing temporal/frequency information. / 位元細化振幅，不能補回流失時間／頻率資訊。 |
| Why context in output? / 為何輸出也要情境？ | Perception depends on listener/viewer state and environment, not just numerical samples. / 知覺取決於使用者狀態及環境，不只是樣本數值。 |

### 8.4 Frequent Mistakes / 常見失誤

**EN:** Avoid writing that 0 dB means no sound; that 16-bit audio has 16 levels; that 44.1 kHz is the audible pitch of a recording; that 5.1 means six identical full-range channels; that all unvoiced sounds have no resonances; that face detection establishes identity; or that more pixels eliminate the semantic gap. Each statement confuses a representation, measurement, or processing stage with another.

**繁中：** 不要寫成：0 dB 等於沒有聲音；16 位元只有 16 級；44.1 kHz 是錄音的可聽音高；5.1 是六個完全相同的全頻聲道；所有無聲語音都沒有共振；人臉偵測已確立身份；增加像素便能消除語意鴻溝。這些都把不同表示、量度或處理階段混為一談。

<a id="glossary"></a>
## 9. Bilingual Terminology / 雙語詞彙表

| English | 繁體中文 |
|---|---|
| Multimedia computing | 多媒體運算 |
| Modality | 模態／感官媒態 |
| Data stream | 資料流 |
| Synchronization | 同步 |
| Feature extraction | 特徵抽取 |
| Metadata | 中繼資料／元資料 |
| Content | 內容 |
| Context | 情境／脈絡 |
| Experience | 經驗／體驗 |
| Information | 資訊 |
| Knowledge | 知識 |
| Insight | 洞見 |
| Experiential environment | 體驗式環境 |
| Human-machine synergy | 人機協同 |
| Perception | 知覺 |
| Percept | 知覺單元 |
| Schema | 基模 |
| Perceptual cycle | 知覺循環 |
| Semantic gap | 語意鴻溝 |
| Object / event | 物件／事件 |
| Sensor / actuator | 感測器／致動器 |
| Dynamic range | 動態範圍 |
| Bias / offset | 偏差／偏移 |
| Nonlinearity | 非線性 |
| Hysteresis / drift | 遲滯／漂移 |
| Sinusoid | 正弦波 |
| Amplitude / phase | 振幅／相位 |
| Frequency / period | 頻率／週期 |
| Sampling / sampling rate | 取樣／取樣率 |
| Quantization / bit depth | 量化／位元深度 |
| Quantization noise | 量化噪音 |
| Nyquist frequency | 奈奎斯特頻率 |
| Nyquist rate | 奈奎斯特取樣率 |
| Aliasing | 混疊 |
| Anti-aliasing filter | 抗混疊濾波器 |
| Root mean square (RMS) | 均方根 |
| Sound pressure level (SPL) | 聲壓級 |
| A-weighting | A 加權 |
| Echo / reverberation | 回聲／混響 |
| Constructive / destructive interference | 相長／相消干涉 |
| Polar pattern / cardioid | 指向圖／心形指向 |
| Frequency response | 頻率響應 |
| Gain / attenuation | 增益／衰減 |
| Clipping / headroom | 削波／餘量 |
| Active noise cancellation (ANC) | 主動降噪 |
| Equalizer | 等化器 |
| Spectrogram / spectrum | 聲譜圖／頻譜 |
| Audio fingerprint | 聲音指紋 |
| Phoneme | 音素 |
| Voiced / unvoiced | 有聲／無聲 |
| Fundamental frequency / pitch | 基頻／音高 |
| Formant | 共振峰 |
| Harmonic / overtone / partial | 諧波／泛音／分音 |
| Timbre / envelope | 音色／包絡 |
| Attack / decay / sustain / release | 起音／衰減／持續／釋音 |
| Vocoder / LPC | 聲碼器／線性預測編碼 |
| Frequency / temporal masking | 頻率／時間遮蔽 |
| Lossy / lossless compression | 有損／無損壓縮 |
| Enhancement / stabilization / deblurring | 增強／防震／去模糊 |
| Segmentation / tracking | 分割／追蹤 |
| Detection / recognition / verification | 偵測／辨識／驗證 |
| Mesh / vertex / facet | 網格／頂點／面 |
| Pose / texture map | 姿態／紋理貼圖 |
| Gait / silhouette / codeword | 步態／輪廓／碼字 |
| Indexing / retrieval | 建立索引／檢索 |
| Crossover / enclosure | 分頻／音箱 |
| Interaural level difference (ILD) | 雙耳聲級差 |
| Interaural time difference (ITD) | 雙耳時間差 |
| Head-related transfer function (HRTF) | 頭部相關傳遞函數 |
| Coherence / polarization | 相干性／偏振 |
| Reflection / refraction | 反射／折射 |
| Luminous intensity / candela | 發光強度／坎德拉 |
| Luminous flux / lumen | 光通量／流明 |
| Illuminance / lux | 照度／勒克斯 |
| Solid angle / steradian | 立體角／球面度 |
| Focal length / image plane | 焦距／影像平面 |
| Frame rate / exposure | 影格率／曝光 |
| Stereo / disparity / baseline | 立體視覺／視差／基線 |
| Time-of-Flight (ToF) | 飛行時間 |
| Folded optics | 摺疊光學 |
| Additive / subtractive colour | 加色法／減色法 |
| Luma / chroma | 亮度相關分量／色度 |
| Gamut / white point | 色域／白點 |
| Autostereoscopic display | 裸眼立體顯示器 |
| Cone / rod / fovea | 視錐／視桿／中央凹 |
| Photopic / scotopic vision | 明視覺／暗視覺 |
| Brightness adaptation | 亮度適應 |

<a id="coverage"></a>
## 10. Source Coverage Map / 來源涵蓋表

**EN:** Use this map to return to the original figures. Title pages and repeated demonstrations are grouped with the surrounding topic; the guide follows the conceptual content rather than duplicating every slide. Linked external videos and product pages are not treated as extra syllabus material, and no unseen demonstration results are assumed.

**繁中：** 可用此表返回原始圖表。封面及重複示範歸入附近主題；筆記按概念內容整理，並非逐張複製投影片。外部影片及產品網頁不被視為額外課綱，也不假設未觀看示範的結果。

| Source / 來源 | PDF page coverage / PDF 頁次涵蓋 | Notes / 筆記位置 |
|---|---|---|
| L01 | 1: title; 2-8: multimedia; 9-16: communication; 17-24: computing history; 25-27: emerging applications; 28-32: experiential environments; 33-35: formal streams and trip example. / 1：封面；2-8：多媒體；9-16：通訊；17-24：電腦發展；25-27：新興應用；28-32：體驗式環境；33-35：資料流模型及旅行例子。 | Sections 1.1-1.7; Q1-Q4 / 第 1.1 至 1.7 節；Q1 至 Q4 |
| L02 | 1-6: objective and experience; 7-8: dictionaries, objects/events; 9-16: perception, ambiguity, cycle; 17-19: system problems and semantic gap; 20-21: context; 22: components. / 1-6：目標及經驗；7-8：字典、物件／事件；9-16：知覺、歧義及循環；17-19：系統問題及語意鴻溝；20-21：情境；22：組件。 | Sections 2.1-2.10; Q5-Q6 / 第 2.1 至 2.10 節；Q5 至 Q6 |
| L03 | 1-21: audio applications; 22-30: enhancement and motion; 31-34: detection/tracking; 35-46: 3D faces; 47-49: recognition limits/Face ID; 50-54: gait; 55-59: retrieval; 60-63: metaverse, DIKW, analytics, smart city. / 1-21：聲音應用；22-30：增強與動作；31-34：偵測／追蹤；35-46：三維人臉；47-49：辨識限制／Face ID；50-54：步態；55-59：檢索；60-63：元宇宙、DIKW、分析及智慧城市。 | Sections 3.1-3.16; W10; Q7-Q12 / 第 3.1 至 3.16 節；W10；Q7 至 Q12 |
| L04 | 1-7: sensors/touch; 8-15: vision demonstrations; 16-19: hearing and other senses; 20-22: life/environment; 23-24: limitations; 25-30: signals; 31-34: digitization/Nyquist; 35-40: acoustics; 41-49: recording and microphones. / 1-7：感測器／觸覺；8-15：視覺示範；16-19：聽覺及其他感官；20-22：生命／環境；23-24：限制；25-30：訊號；31-34：數碼化／奈奎斯特；35-40：聲學；41-49：錄音及麥克風。 | Sections 4.1-4.16; W1-W9; Q13-Q17 / 第 4.1 至 4.16 節；W1 至 W9；Q13 至 Q17 |
| L05 | 1-8: speakers/spatial audio; 9-15: speech; 16-20: music; 21-29: light/units; 30-36: optics/formation; 37-40: sensors/resolution; 41-46: depth and optical devices; 47-52: reproduction/displays; 53-59: vision; 60-62: colour and True Tone. / 1-8：揚聲器／空間聲音；9-15：語音；16-20：音樂；21-29：光／單位；30-36：光學／成像；37-40：感測器／解像度；41-46：深度及光學裝置；47-52：重現／顯示；53-59：視覺；60-62：色彩及 True Tone。 | Sections 5.1-5.25; W11-W15; Q18-Q24 / 第 5.1 至 5.25 節；W11 至 W15；Q18 至 Q24 |

**Final principle / 最後原則:** Start with the problem and the information needed, then choose sensors, representations, integration, and presentation. Do not start by collecting media and assume meaning will appear automatically. / 先確定問題及所需資訊，再選擇感測器、表示、整合及呈現方法；不要先堆積媒體，再假設意義會自動出現。