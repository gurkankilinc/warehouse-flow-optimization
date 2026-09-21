"""Turkish / English strings for the dashboard.

Kept in one table so a missing translation is obvious rather than silently
falling back to a half-translated page. ``t(key)`` raises in neither language:
an unknown key returns the key itself, which shows up loudly in the UI.
"""

from __future__ import annotations

LANGUAGES = {"tr": "Türkçe", "en": "English"}

STRINGS: dict[str, dict[str, str]] = {
    # ---------------- shell ----------------
    "app_title": {
        "tr": "Depo Akış Optimizasyonu",
        "en": "Warehouse Flow Optimization",
    },
    "app_subtitle": {
        "tr": "Tahmine dayalı raf yerleşimi, teslim süresi bilinçli sevkiyat ve toplama "
              "rotası — bir dağıtım merkezinin olay simülasyonu üzerinde ölçüldü.",
        "en": "Forecast-driven slotting, deadline-aware dispatch and pick routing — "
              "measured on a discrete-event simulation of a distribution centre.",
    },
    "language": {"tr": "Dil", "en": "Language"},
    "sidebar_facility": {"tr": "Tesis", "en": "Facility"},
    "slots": {"tr": "Raf gözü", "en": "Storage slots"},
    "skus": {"tr": "Ürün çeşidi (SKU)", "en": "SKUs"},
    "orders_window": {"tr": "Sipariş (pencere)", "en": "Orders (window)"},
    "occupancy": {"tr": "Doluluk", "en": "Occupancy"},
    "calibration_source": {"tr": "Talep kalibrasyon kaynağı", "en": "Demand calibration source"},
    "artifacts_missing": {
        "tr": "Eğitilmiş model dosyaları bulunamadı. Şunu çalıştırın: "
              "`python -m src.experiments.run_comparison`",
        "en": "Trained artefacts not found. Run: "
              "`python -m src.experiments.run_comparison`",
    },
    "no_results_yet": {
        "tr": "Henüz deney sonucu yok. Şunu çalıştırın: "
              "`python -m src.experiments.run_comparison`",
        "en": "No experiment results yet. Run: "
              "`python -m src.experiments.run_comparison`",
    },

    # ---------------- tabs ----------------
    "tab_overview": {"tr": "Genel Bakış", "en": "Overview"},
    "tab_warehouse": {"tr": "Depo", "en": "Warehouse"},
    "tab_map": {"tr": "Optimizasyon Haritası", "en": "Optimization Map"},
    "tab_journey": {"tr": "Sipariş Yolculuğu", "en": "Order Journey"},
    "tab_putaway": {"tr": "Mal Kabul", "en": "Putaway"},
    "tab_results": {"tr": "Sonuçlar", "en": "Results"},

    # ---------------- overview ----------------
    "ov_problem_h": {"tr": "Problem", "en": "The problem"},
    "ov_problem": {
        "tr": "Bir dağıtım merkezi her gün iki soru sorar. **Çıkışta:** elindeki teslim "
              "süreli siparişleri hangi sırayla, hangi gruplamayla ve hangi rotayla "
              "toplayacak? **Girişte:** mal kabule gelen paleti hangi rafa koyarsa, o "
              "ürünün önümüzdeki haftalarda üreteceği toplama işi en ucuz olacak?\n\n"
              "Birincisi çevrimiçi bir çizelgeleme, ikincisi kısıtlı bir atama problemi. "
              "İkisini birbirine bağlayan şey bir tahmindir: neyin toplanacağını "
              "öngörmeden stoğu iyi yerleştiremezsiniz.",
        "en": "A distribution centre asks two questions every day. **Outbound:** which "
              "order gets picked next, batched with what, and in what sequence are the "
              "stops visited? **Inbound:** a pallet arrives at goods-in — which slot "
              "should it go to, so the picks it generates over the coming weeks are as "
              "cheap as possible?\n\n"
              "The first is an online scheduling problem, the second a constrained "
              "assignment problem. What links them is a forecast: you cannot place stock "
              "well without a view of what will be picked.",
    },
    "ov_goal_h": {"tr": "Hedef", "en": "The goal"},
    "ov_goal": {
        "tr": "Tek bir sayıyı iyileştirmek: **siparişlerin yüzde kaçı kendisine söz "
              "verilen kamyona yetişiyor.** Yürüme mesafesi, tur sayısı ve toplayıcı "
              "doluluğu bunun altındaki ara metriklerdir.\n\n"
              "Karşılaştırma tabanı zayıf seçilmedi: rastgele bir yerleşimi yenmek hiçbir "
              "şey kanıtlamaz. Taban, yetkin depoların gerçekten kullandığı yöntemler — "
              "ABC yerleştirme, EDD sevkiyat ve S-shape rota.",
        "en": "To move one number: **what share of orders leave on the truck they were "
              "promised.** Walking distance, tour count and picker utilisation are the "
              "intermediate metrics underneath it.\n\n"
              "The baseline is not a strawman — beating a random layout would prove "
              "nothing. It is what competent warehouses actually do: ABC slotting, "
              "earliest-due-date dispatch and S-shape routing.",
    },
    "ov_solution_h": {"tr": "Çözüm", "en": "The solution"},
    "ov_solution": {
        "tr": "Gerçekten tahmin gereken yerde makine öğrenmesi, kararın yapısı olan yerde "
              "klasik optimizasyon. Pekiştirmeli öğrenme değerlendirildi ve elendi: durum "
              "uzayı çok büyük, ödül seyrek ve gecikmeli, ve ortaya çıkan politikayı "
              "onaylayacak operasyon müdürüne açıklamak neredeyse imkânsız.",
        "en": "Machine learning where a prediction is genuinely needed, classical "
              "optimization where the decision has structure worth exploiting. "
              "Reinforcement learning was considered and rejected: the state space is "
              "enormous, the reward sparse and delayed, and the resulting policy is "
              "nearly impossible to explain to the operations manager who has to approve "
              "it.",
    },
    "ov_headline_h": {"tr": "Sonuç", "en": "The outcome"},
    "ov_reco_h": {"tr": "Öneriler", "en": "Recommendations"},
    "ov_reco": {
        "tr": "Ölçümlerin işaret ettiği eylem sırası — en yüksek getiriden en düşüğe:\n\n"
              "1. **Sevkiyat kuralını değiştirin (triage).** Kod değişikliği küçük, "
              "kazanç en büyük: kaçırılan kamyonlarda tek başına %86 azalma. Yeni veri, "
              "yeni model veya yeni donanım gerektirmez.\n"
              "2. **Çapraz koridorlarınızı sayın.** Tek çapraz koridorlu bir blokta rota "
              "optimizasyonu matematiksel olarak %0 kazandırır. Bu bir yazılım değil, "
              "yerleşim planı kararıdır.\n"
              "3. **Gruplama ve rotayı birlikte devreye alın.** Yürüme mesafesinde %7 "
              "kazanç; triage'ın gruplamayı bozmasından doğan kaybı da telafi eder.\n"
              "4. **Yerleştirmeyi en sona bırakın.** Güçlü bir ABC tabanına karşı ölçülen "
              "kazanç küçük ve istatistiksel olarak belirsiz. Stok taşıma maliyeti "
              "modellenmeden yatırım kararı verilmemeli.\n"
              "5. **SLA risk skorunu sevkiyat önceliğine bağlamayın.** Ölçüm sonucu "
              "kötüleştiriyor. Skoru operatöre *uyarı* olarak göstermek makul olabilir, "
              "ama sıraya otomatik müdahale ettirmek değil.",
        "en": "The order of action the measurements point to — highest return first:\n\n"
              "1. **Change the dispatch rule (triage).** Small code change, largest gain: "
              "86% fewer missed trucks on its own. Needs no new data, no model and no "
              "hardware.\n"
              "2. **Count your cross aisles.** In a single-block layout, routing "
              "optimisation is mathematically worth 0%. That is a layout decision, not a "
              "software one.\n"
              "3. **Deploy batching and routing together.** 7% off walking distance, and "
              "it repays the batching that triage breaks up.\n"
              "4. **Leave slotting until last.** Against a strong ABC baseline the "
              "measured gain is small and statistically uncertain. Do not commit to it "
              "without first modelling what relocating stock costs.\n"
              "5. **Do not wire the SLA risk score into dispatch priority.** It measurably "
              "makes things worse. Showing the score to an operator as a *warning* may be "
              "reasonable; letting it reorder the queue automatically is not.",
    },
    "ov_honest_h": {"tr": "İşe yaramayanlar", "en": "What did not work"},
    "ov_honest": {
        "tr": "Üç modelden ikisi yerini hak etmedi ve ikisi de kanıtıyla birlikte repoda "
              "duruyor:\n\n"
              "- **Toplama süresi tahmini:** doğrusal regresyon gradient boosting'e eşit "
              "(MAE 183,1 sn / 183,8 sn). Sevkiyat kuralına takıldığında zamanında sevk "
              "oranını dört ondalık basamağa kadar değiştirmiyor.\n"
              "- **SLA risk eskalasyonu:** karar-içi AUC'de slack'i açık ara yeniyor "
              "(0,82 / 0,62) ama average precision'ı üçte biri. Sıralamanın gövdesini iyi, "
              "tepesini kötü sıralıyor — eskalasyon ise sadece tepeyi harcar.",
        "en": "Two of the three models did not earn their place, and both remain in the "
              "repository with the evidence:\n\n"
              "- **Pick-time prediction:** a linear regression matches gradient boosting "
              "(MAE 183.1 s / 183.8 s). Wired into the dispatch rule it changes the "
              "on-time rate by nothing to four decimal places.\n"
              "- **SLA-risk escalation:** it clearly beats slack on within-decision AUC "
              "(0.82 / 0.62) but has a third of its average precision. It ranks the body "
              "of the queue well and the top badly — and escalation spends only the top.",
    },

    # ---------------- warehouse ----------------
    "wh_layout_h": {"tr": "Tesis planı", "en": "Facility layout"},
    "wh_specs_h": {"tr": "Depo özellikleri", "en": "Warehouse specification"},
    "wh_zones_h": {"tr": "Depolama bölgeleri ve kısıtlar", "en": "Storage zones and constraints"},
    "wh_heat_h": {"tr": "Talep yoğunluğu haritası", "en": "Demand intensity map"},
    "wh_heat_help": {
        "tr": "Her dikdörtgen bir raf gözü sütunu (üç kat birlikte). Renk, o gözde duran "
              "ürünün ölçüm penceresinde kaç kez toplandığını gösterir. İyi bir yerleşimde "
              "sıcak renkler dock'un çevresinde toplanır.",
        "en": "Each rectangle is one rack column (its three levels combined). Colour is how "
              "many times the SKU stored there was picked during the measured window. In a "
              "good layout the hot colours cluster around the dock.",
    },
    "wh_bands_h": {"tr": "Mesafe bandına göre toplama işi", "en": "Pick work by distance band"},
    "wh_bands_help": {
        "tr": "Isı haritası nerede olduğunu gösterir, bu grafik ne kadar olduğunu. Toplama "
              "işinin ne kadarının dock'a yakın bantlarda gerçekleştiği, yerleşimin "
              "kalitesinin tek satırlık özetidir.",
        "en": "The heat map shows where; this shows how much. What share of picking work "
              "happens in the bands nearest the dock is the one-line summary of layout "
              "quality.",
    },
    "wh_abcxyz_h": {"tr": "ABC × XYZ segmentasyonu", "en": "ABC x XYZ segmentation"},
    "wh_abcxyz_help": {
        "tr": "ABC ürünleri hacme, XYZ ise haftalık talebin ne kadar öngörülebilir olduğuna "
              "göre ayırır. Köşegen desen gerçek envanter analizlerinde beklenen sonuçtur: "
              "çok satanlar aynı zamanda istikrarlı olanlardır, uzun kuyruk ise düzensiz.",
        "en": "ABC splits products by volume, XYZ by how predictable their weekly demand "
              "is. The diagonal is what real inventory analyses show: the big sellers are "
              "also the steady ones, and the long tail is erratic.",
    },

    # ---------------- optimization map ----------------
    "map_h": {"tr": "Neyi, nerede, nasıl optimize ediyoruz", "en": "What we optimize, where, and how"},
    "map_intro": {
        "tr": "Sistem üç farklı zaman ölçeğinde karar veriyor. Her katman farklı bir girdiyi "
              "kullanır, farklı bir yöntemle çözülür ve farklı bir maliyeti düşürür. "
              "Aşağıdaki harita bu üç katmanı ve aralarındaki bağı gösteriyor.",
        "en": "The system decides on three different time scales. Each layer consumes a "
              "different input, is solved by a different method, and reduces a different "
              "cost. The map below shows the three layers and how they connect.",
    },
    "map_flow_h": {"tr": "Bir siparişin geçtiği aşamalar", "en": "The stages an order passes through"},
    "map_table_h": {"tr": "Katman katman karşılaştırma", "en": "Layer by layer"},
    "map_col_layer": {"tr": "Katman", "en": "Layer"},
    "map_col_horizon": {"tr": "Ölçek", "en": "Horizon"},
    "map_col_decision": {"tr": "Karar", "en": "Decision"},
    "map_col_baseline": {"tr": "Taban yöntem", "en": "Baseline method"},
    "map_col_optimized": {"tr": "Optimize yöntem", "en": "Optimized method"},
    "map_col_gain": {"tr": "Ölçülen kazanç", "en": "Measured gain"},

    # ---------------- journey ----------------
    "jr_h": {"tr": "Bir ürün depodan nasıl en hızlı çıkar?", "en": "How does a product leave the warehouse fastest?"},
    "jr_intro": {
        "tr": "Bir sipariş seçin. Aşağıda o siparişin fiziksel yolculuğunu adım adım "
              "göreceksiniz: hangi raflara gidiliyor, hangi sırayla, ne kadar yürünüyor ve "
              "hangi kamyona yetişmesi gerekiyor.",
        "en": "Pick an order. Below is its physical journey step by step: which racks are "
              "visited, in what sequence, how far the picker walks, and which truck it has "
              "to catch.",
    },
    "jr_select_order": {"tr": "Sipariş", "en": "Order"},
    "jr_service": {"tr": "Servis seviyesi", "en": "Service level"},
    "jr_lines": {"tr": "Kalem", "en": "Lines"},
    "jr_units": {"tr": "Adet", "en": "Units"},
    "jr_deadline": {"tr": "Son hazırlık anı", "en": "Staging deadline"},
    "jr_route_h": {"tr": "Toplama rotası", "en": "Picking route"},
    "jr_route_help": {
        "tr": "Aynı duraklar, iki farklı sırayla. S-shape koridorları sırayla tarar; "
              "NN + 2-opt çapraz koridorları kestirme olarak kullanabilir. Numaralar "
              "ziyaret sırasını gösterir.",
        "en": "The same stops, sequenced two ways. S-shape sweeps the aisles in order; "
              "NN + 2-opt is free to use the cross aisles as shortcuts. The numbers are "
              "the visiting order.",
    },
    "jr_timeline_h": {"tr": "Zaman çizelgesi", "en": "Timeline"},
    "jr_timeline_help": {
        "tr": "Teslim süresi uydurma bir sayı değil: siparişin bindiği kamyonun planlanan "
              "kalkışından hazırlık tamponu düşülerek hesaplanır.",
        "en": "The deadline is not an invented number: it is the planned departure of the "
              "truck the order is assigned to, minus the staging buffer.",
    },
    "jr_stops_h": {"tr": "Duraklar", "en": "Stops"},
    "jr_saving": {"tr": "Rota kazancı", "en": "Route saving"},
    "jr_no_orders": {"tr": "Çizime uygun boyutta sipariş bulunamadı.", "en": "No suitably sized order to draw."},

    # ---------------- putaway ----------------
    "pa_h": {"tr": "Palet geldi. Nereye konmalı?", "en": "A pallet arrived. Where should it go?"},
    "pa_intro": {
        "tr": "Öneri, ürünün kullanmasına izin verilen en yakın raf gözüdür — ne sıklıkla "
              "toplanacağı ve bu toplamaların ne kadar acil olduğu ile ağırlıklandırılır.",
        "en": "The recommendation is the nearest slot the product is allowed to occupy, "
              "weighted by how often it is expected to be picked and how urgent those "
              "picks are.",
    },
    "pa_sku": {"tr": "Gelen ürün", "en": "Incoming SKU"},
    "pa_category": {"tr": "Kategori", "en": "Category"},
    "pa_forecast": {"tr": "Tahmini toplama / pencere", "en": "Forecast picks / window"},
    "pa_urgency": {"tr": "Aynı gün payı", "en": "Same-day share"},
    "pa_constraints": {"tr": "Kısıtlar", "en": "Constraints"},
    "pa_none": {"tr": "yok", "en": "none"},
    "pa_cold": {"tr": "soğuk zincir", "en": "cold chain"},
    "pa_heavy": {"tr": "ağır (zemin kat)", "en": "heavy (ground level)"},
    "pa_recommended": {"tr": "Önerilen göz", "en": "Recommended slot"},
    "pa_baseline_slot": {"tr": "ABC tabanının seçeceği göz", "en": "Slot the ABC baseline would use"},
    "pa_walk": {"tr": "dock'tan yürüme", "en": "walk from dock"},
    "pa_diff": {"tr": "ziyaret başına fark", "en": "difference per visit"},

    # ---------------- shipping: two halves ----------------
    "tab_placement": {"tr": "1 · Yerleştirme", "en": "1 · Placement"},
    "tab_evacuation": {"tr": "2 · Tahliye", "en": "2 · Evacuation"},

    "two_halves_h": {
        "tr": "Sevkiyat tek bir iş değil, iki ayrı problem",
        "en": "Shipping is not one job, it is two problems",
    },
    "two_halves": {
        "tr": "Bir ürünün depodan çıkması iki bağımsız kararın sonucudur ve ikisi farklı "
              "zaman ölçeğinde, farklı verilerle, farklı kişiler tarafından verilir.\n\n"
              "**1. Yerleştirme — mal girişinde verilir, haftalarca etkisini sürdürür.** "
              "Ürün rampaya yakın bir göze mi, uzağa mı konacak? Bu karar bir kez verilir; "
              "bedeli ise o ürün her toplandığında yeniden ödenir.\n\n"
              "**2. Tahliye — sevkiyat anında verilir, dakikalar içinde biter.** Araç "
              "kapıda; hangi sipariş önce toplanacak, neyle gruplanacak, hangi rotayla "
              "gezilecek, ve hazırlanan mal araca hangi sırayla yüklenecek?\n\n"
              "Bu ikisini ayırmak sadece anlatım kolaylığı değil: ölçüm, kaçırılan her "
              "kamyonu bu iki taraftan birine yazıyor — ve sonuç, hangisine yatırım "
              "yapmanız gerektiğini doğrudan söylüyor.",
        "en": "A product leaving the building is the result of two independent decisions, "
              "made on different time scales, from different data, by different people.\n\n"
              "**1. Placement — decided at goods-in, felt for weeks.** Does this product go "
              "in a slot near the ramp or far from it? The decision is made once; the bill "
              "is paid again on every single pick.\n\n"
              "**2. Evacuation — decided at the moment of shipping, over in minutes.** The "
              "vehicle is at the door: which order is picked first, batched with what, "
              "routed how, and in what order does the staged stock go onto the deck?\n\n"
              "Separating them is not a presentational convenience. The measurement "
              "attributes every missed truck to one side or the other — and the answer "
              "tells you directly which one is worth investing in.",
    },

    # --- placement tab ---
    "pl_h": {
        "tr": "Rampaya yakın ne konur, uzağa ne konur?",
        "en": "What goes near the ramp, and what goes far?",
    },
    "pl_intro": {
        "tr": "Bir raf gözünün tek bir maliyeti vardır: dock'a olan yürüme mesafesi. Bir "
              "ürünün tek bir değeri vardır: o mesafeyi kaç kez yürüteceği. Karar kuralı "
              "bu ikisini çarpıp toplamı en aza indirmekten ibarettir.\n\n"
              "Kural şu: **skor = beklenen toplama sayısı × (1 + β × aynı-gün payı)**. "
              "En yüksek skorlu ürün, kullanmasına izin verilen en yakın gözü alır.",
        "en": "A slot has exactly one cost: its walking distance from the dock. A product "
              "has exactly one value: how many times it will make someone walk it. The "
              "decision rule multiplies the two and minimises the sum.\n\n"
              "The rule is **score = expected picks x (1 + beta x same-day share)**. The "
              "highest-scoring product takes the nearest slot it is permitted to occupy.",
    },
    "pl_why_h": {"tr": "Neden bu ürün oraya gitti?", "en": "Why did this product go there?"},
    "pl_beta_h": {"tr": "Aciliyet ağırlığı (β) ne yapıyor?", "en": "What does the urgency weight (beta) do?"},
    "pl_beta_help": {
        "tr": "β = 0 iken kural saf sıklığa döner (klasik ABC). β büyüdükçe, seyrek "
              "satılan ama hep aynı gün çıkması gereken ürünler öne geçer. Kaydırıcıyı "
              "oynatın: kimin kazanıp kimin kaybettiğini canlı görün.",
        "en": "At beta = 0 the rule collapses to pure frequency (classic ABC). As beta "
              "grows, products that sell rarely but always ship same-day move forward. "
              "Move the slider and watch who gains and who loses.",
    },
    "pl_near_h": {"tr": "Rampaya en yakın 15 ürün", "en": "The 15 products nearest the ramp"},
    "pl_far_h": {"tr": "Rampadan en uzak 15 ürün", "en": "The 15 products farthest from the ramp"},
    "pl_profile_h": {"tr": "Mesafeye göre ürün profili", "en": "Product profile by distance"},
    "pl_profile_help": {
        "tr": "Dock'a yaklaştıkça hangi sınıfların yoğunlaştığı, kuralın gerçekten "
              "çalışıp çalışmadığının en doğrudan kanıtıdır.",
        "en": "Which classes concentrate as you approach the dock is the most direct "
              "evidence of whether the rule is actually working.",
    },
    "pl_score": {"tr": "Skor", "en": "Score"},
    "pl_rank": {"tr": "Sıra", "en": "Rank"},
    "pl_forecast_picks": {"tr": "Tahmini toplama", "en": "Forecast picks"},
    "pl_urgency_w": {"tr": "Aciliyet ağırlığı", "en": "Urgency weight"},
    "pl_assigned_dist": {"tr": "Atanan mesafe", "en": "Assigned distance"},

    # --- evacuation tab ---
    "ev_h": {
        "tr": "Araç kapıda: ürünü bulup yüklemek",
        "en": "The vehicle is at the door: find it and load it",
    },
    "ev_intro": {
        "tr": "Tahliye üç aşamadır ve her biri ayrı bir kaynakla sınırlıdır: **toplama** "
              "(toplayıcılar), **hazırlama alanında bekleme** (alan kapasitesi), ve "
              "**araca yükleme** (yükleme ekibi ve kalkışa kalan süre).\n\n"
              "Bir sipariş kamyonunu iki farklı sebeple kaçırabilir: ya zamanında "
              "toplanamamıştır, ya da hazır olduğu hâlde ekip ona sıra getirememiştir. "
              "Bu ikisini ayırmak, hangi tarafa kapasite ekleneceğini söyleyen tek sayıdır.",
        "en": "Evacuation is three stages, each limited by a different resource: "
              "**picking** (the pickers), **waiting on the staging lanes** (their "
              "capacity), and **loading onto the vehicle** (the crew, and the time left "
              "before departure).\n\n"
              "An order can miss its truck for two different reasons: it was not picked "
              "in time, or it was ready and the crew never got to it. Separating those is "
              "the single number that says which side needs capacity.",
    },
    "ev_attrib_h": {"tr": "Kaçırılan kamyonlar neyin yüzünden?", "en": "What causes the missed trucks?"},
    "ev_pipeline_h": {"tr": "Sipariş hattı: her aşamada geçen süre", "en": "Order pipeline: time spent at each stage"},
    "ev_truck_h": {"tr": "Araç bazında yükleme", "en": "Loading, truck by truck"},
    "ev_truck_help": {
        "tr": "Her nokta bir araç. Yatay eksen, kalkışa kadar ekibin elindeki sürenin "
              "yüzde kaçını kullandığını gösterir. %100'e dayanan araçlar, hazır malı "
              "geride bırakarak kalkanlardır.",
        "en": "Each point is one vehicle. The horizontal axis is how much of the crew's "
              "available time before departure was actually used. Vehicles pressed "
              "against 100% are the ones that left ready stock behind.",
    },
    "ev_release_h": {"tr": "Dalga bırakma: iki yarıyı birbirine bağlayan kural", "en": "Wave release: the rule that joins the two halves"},
    "ev_release_help": {
        "tr": "Kamyonu henüz uzakta olan sipariş toplamaya açılmaz. Bu kural olmadan "
              "toplayıcılar yarının işini önden toplayıp hazırlama alanını doldurur ve "
              "kapıdaki araca ait işi yapamaz hâle gelir — simülasyon bu kural olmadan "
              "tamamen kilitleniyor.",
        "en": "An order is not released for picking until its truck is close. Without "
              "this rule pickers run ahead on tomorrow's work, fill the staging lanes and "
              "lock themselves out of the work for the vehicle at the door — the "
              "simulation deadlocks outright without it.",
    },
    "ev_staging_dwell": {"tr": "Hazırlama alanında ortalama bekleme", "en": "Mean wait on staging lanes"},
    "ev_window_used": {"tr": "Yükleme penceresi kullanımı", "en": "Loading window used"},
    "ev_orders_per_truck": {"tr": "Araç başına sipariş", "en": "Orders per vehicle"},
    "ev_miss_picking": {"tr": "Toplama kaynaklı", "en": "Caused by picking"},
    "ev_miss_loading": {"tr": "Yükleme kaynaklı", "en": "Caused by loading"},
    "ev_ceiling_h": {"tr": "Bu neyi sınırlıyor?", "en": "What does this cap?"},

    # ---------------- results ----------------
    "rs_h": {"tr": "Gerçekten işe yarıyor mu?", "en": "Does it actually help?"},
    "rs_headline_h": {"tr": "Manşet", "en": "Headline"},
    "rs_missed": {"tr": "Kamyonunu kaçıran sipariş", "en": "Orders missing their truck"},
    "rs_baseline": {"tr": "Taban", "en": "Baseline"},
    "rs_best": {"tr": "En iyi senaryo", "en": "Best scenario"},
    "rs_distance": {"tr": "Satır başına yürüme", "en": "Walking distance per line"},
    "rs_table_h": {"tr": "Senaryo karşılaştırması", "en": "Scenario comparison"},
    "rs_table_help": {
        "tr": "Replikasyonlar üzerinden ortalama. Bileşenler ayrı ayrı da koşuldu, böylece "
              "kazanç iddia edilmek yerine atfedilebiliyor.",
        "en": "Mean over replications. Each component was also run on its own, so the gain "
              "can be attributed rather than asserted.",
    },
    "rs_delta_h": {"tr": "Tabana göre eşleştirilmiş fark", "en": "Paired difference against the baseline"},
    "rs_delta_help": {
        "tr": "Her replikasyon bütün senaryolarda aynı tıkanıklık, toplayıcı becerisi ve "
              "tır gecikmesi gerçekleşmesini gördü. Dolayısıyla farklar eşleştirilmiştir ve "
              "aralık, seviyenin değil farkın yayılımıdır.",
        "en": "Each replication faced identical congestion, picker skill and truck delays "
              "under every scenario, so these differences are paired and the interval is "
              "the spread of the difference, not of the level.",
    },
    "rs_capacity_h": {"tr": "Kapasite: bir depo müdürünün soracağı sayı", "en": "Capacity: the number a warehouse manager asks for"},
    "rs_capacity_help": {
        "tr": "Dikey farkı okursanız bu bir servis seviyesi iyileştirmesi; yatay farkı "
              "okursanız bir kapasite iyileştirmesi. Bütçesi olan çerçeve ikincisidir.",
        "en": "Read the vertical gap and this is a service-level improvement; read the "
              "horizontal gap and it is a capacity one. The second framing is the one with "
              "a budget attached.",
    },
    "rs_models_h": {"tr": "Model karneleri", "en": "Model scorecards"},
    "rs_models_help": {
        "tr": "Her model, yenmesi gereken kuralın yanında raporlanıyor. Tek başına bir hata "
              "metriği, modelin kurulmaya değip değmediği hakkında hiçbir şey söylemez.",
        "en": "Every model is reported next to the rule it has to beat. An error metric on "
              "its own says nothing about whether the model was worth building.",
    },
    "rs_stress_h": {"tr": "Stres testleri", "en": "Stress tests"},
    "rs_stress_help": {
        "tr": "Modeller, tahmin ve raf yerleşimi normal koşullarda eğitildiği gibi kalıyor — "
              "hiçbiri yeniden eğitilmiyor. Sahaya alınmış bir sistemin karşılaştığı durum "
              "da budur.",
        "en": "The models, the forecast and the slot assignment stay exactly as fitted "
              "under normal conditions — nothing is refitted. That is what a deployed "
              "system faces.",
    },
    "orders_per_day": {"tr": "Sipariş / gün", "en": "Orders per day"},
    "on_time_rate": {"tr": "Zamanında sevk oranı", "en": "On-time rate"},
}


def t(key: str, lang: str) -> str:
    """Look up a string; unknown keys surface as themselves rather than crashing."""
    entry = STRINGS.get(key)
    if entry is None:
        return f"[{key}]"
    return entry.get(lang) or entry.get("en") or f"[{key}]"
