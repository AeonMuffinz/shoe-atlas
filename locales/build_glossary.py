"""Generates locales/glossary.json.

Run it to regenerate the draft; after that the JSON is the file people edit, especially for
Turkish display names, which are left blank here rather than invented.
"""

from __future__ import annotations

from pathlib import Path

from src.glossary import (
    REVIEW_CONFIRMED,
    REVIEW_SELF,
    TIER_DECODED,
    TIER_DESCRIPTIVE,
    TIER_DIRECT,
    Entry,
    Glossary,
)

INSOLE_REASON: str = (
    "not distinguishable by prompt: Poron, EVA, polyurethane, gel and memory foam look alike even "
    "when the footbed is fully in frame, which it often is on sandals and slippers. The exclusion is "
    "about mutual indistinguishability, not about the insole being hidden."
)
MOC_ALGONQUIN_NOTE: str = (
    "Moc Toe and Algonquin are near-identical in a photograph, differing only in whether the seam is "
    "puckered or a flat raised ridge. Their ceiling is set by visual distinguishability, not prompt "
    "quality, so a low score here is not evidence the prompt is wrong."
)
TERRY_NOTE: str = (
    "91% of Terry positives are Slipper Flats, against slippers being 2.6% of the catalog, a 35x "
    "concentration. A correct prediction may reflect recognising a slipper rather than recognising "
    "terry cloth, so report this caveat wherever the label is scored."
)
BABY_NOTE: str = (
    "Prewalker and Firstwalker are both infant shoes and look similar in a product photo; the "
    "difference is sole stiffness, which a photograph shows poorly. Observation, not reviewed."
)


def direct(label: str, prompt: str, display: str, **kwargs: object) -> Entry:
    return Entry(label=label, prompt=prompt, display_en=display, tier=TIER_DIRECT, **kwargs)  # type: ignore[arg-type]


def described(label: str, prompt: str, display: str, **kwargs: object) -> Entry:
    return Entry(label=label, prompt=prompt, display_en=display, tier=TIER_DESCRIPTIVE, **kwargs)  # type: ignore[arg-type]


def decoded(label: str, prompt: str, display: str, **kwargs: object) -> Entry:
    return Entry(label=label, prompt=prompt, display_en=display, tier=TIER_DECODED, **kwargs)  # type: ignore[arg-type]


def build() -> Glossary:
    entries: list[Entry] = [
        direct("Category.Shoes", "a shoe", "Shoes"),
        direct("Category.Boots", "a boot", "Boots"),
        direct("Category.Sandals", "a sandal", "Sandals"),
        direct("Category.Slippers", "a slipper", "Slippers"),

        direct("SubCategory.Oxfords", "an oxford dress shoe with closed lacing", "Oxfords"),
        direct("SubCategory.Loafers", "a slip-on loafer shoe with no laces", "Loafers"),
        direct("SubCategory.Boat.Shoes", "a leather boat shoe with a lace threaded around the collar", "Boat shoes"),
        direct("SubCategory.Clogs.and.Mules", "a backless clog or mule shoe", "Clogs and mules"),
        direct("SubCategory.Flats", "a flat women's shoe with no heel", "Flats"),
        direct("SubCategory.Heels", "a high-heeled women's shoe", "Heels"),
        direct("SubCategory.Sneakers.and.Athletic.Shoes", "a sneaker or athletic trainer", "Sneakers"),
        direct("SubCategory.Flat", "a flat sandal with no heel", "Flat sandals"),
        direct("SubCategory.Heel", "a heeled sandal with a raised heel", "Heeled sandals"),
        direct("SubCategory.Ankle", "an ankle boot with a short shaft ending at the ankle", "Ankle boots"),
        direct("SubCategory.Mid.Calf", "a boot with a shaft reaching the middle of the calf", "Mid-calf boots"),
        direct("SubCategory.Knee.High", "a knee-high boot with a tall shaft reaching the knee", "Knee-high boots"),
        direct("SubCategory.Slipper.Flats", "a soft flat house slipper", "Slipper flats"),
        direct("SubCategory.Prewalker", "a soft infant shoe for a baby who cannot walk yet", "Prewalker", notes=BABY_NOTE),
        direct("SubCategory.Firstwalker", "a small toddler shoe for a child learning to walk", "Firstwalker", notes=BABY_NOTE),

        decoded("HeelHeight.Flat", "a flat shoe with no raised heel", "Flat"),
        decoded("HeelHeight.Under.1in", "a shoe with a very low heel under one inch high", "Under 1 in"),
        decoded("HeelHeight.1in...1.3.4in", "a shoe with a low heel between one and one and three quarter inches high", "1-1.75 in"),
        decoded("HeelHeight.2in...2.3.4in", "a shoe with a mid heel between two and two and three quarter inches high", "2-2.75 in"),
        decoded("HeelHeight.3in...3.3.4in", "a shoe with a high heel between three and three and three quarter inches high", "3-3.75 in"),
        decoded("HeelHeight.4in...4.3.4in", "a shoe with a very high heel between four and four and three quarter inches high", "4-4.75 in"),
        decoded("HeelHeight.5in...over", "a shoe with an extremely high heel of five inches or more", "5 in and over"),

        direct("Gender.Men", "a men's shoe", "Men"),
        direct("Gender.Women", "a women's shoe", "Women"),
        direct("Gender.Boys", "a boy's shoe", "Boys"),
        direct("Gender.Girls", "a girl's shoe", "Girls"),

        direct("Closure.Lace.up", "a shoe fastened with laces threaded through eyelets", "Lace-up"),
        direct("Closure.Slip.On", "a slip-on shoe with no fastening", "Slip-on"),
        direct("Closure.Pull.on", "a boot pulled on by hand with no fastening", "Pull-on"),
        direct("Closure.Buckle", "a shoe fastened with a metal buckle", "Buckle"),
        direct("Closure.Zipper", "a shoe fastened with a zip", "Zipper"),
        described("Closure.Hook.and.Loop", "a shoe fastened with a hook-and-loop velcro strap", "Hook and loop"),
        direct("Closure.Ankle.Strap", "a shoe with a strap fastening around the ankle", "Ankle strap"),
        direct("Closure.Ankle.Wrap", "a sandal with long straps wrapped around the ankle and lower leg", "Ankle wrap"),
        direct("Closure.Sling.Back", "a shoe with an open back and a strap behind the heel", "Slingback"),
        described("Closure.Elastic.Gore", "a boot with elastic gusset panels at the sides of the ankle", "Elastic gore"),
        described("Closure.Bungee", "a shoe fastened with an elastic bungee cord and a spring toggle", "Bungee"),
        described("Closure.Toggle", "a shoe fastened with a toggle clasp", "Toggle"),
        described("Closure.Button.Loop", "a shoe fastened with a button passed through a fabric loop", "Button loop"),
        described("Closure.Monk.Strap", "a monk strap shoe fastened with a wide strap and buckle across the instep instead of laces", "Monk strap"),

        direct("Material.Leather", "a shoe made of smooth leather", "Leather"),
        direct("Material.Suede", "a shoe made of suede with a soft napped surface", "Suede"),
        direct("Material.Canvas", "a shoe made of woven canvas fabric", "Canvas"),
        direct("Material.Rubber", "a shoe made of rubber", "Rubber"),
        direct("Material.Cotton", "a shoe made of cotton fabric", "Cotton"),
        direct("Material.Nylon", "a shoe made of nylon fabric", "Nylon"),
        direct("Material.Mesh", "a shoe made of open mesh fabric with visible perforations", "Mesh"),
        direct("Material.Polyester", "a shoe made of polyester fabric", "Polyester"),
        direct("Material.Synthetic", "a shoe made of synthetic man-made material", "Synthetic"),
        direct("Material.Faux.Leather", "a shoe made of imitation leather", "Faux leather"),
        direct("Material.Patent.Leather", "a shoe made of glossy patent leather with a mirror-like shine", "Patent leather"),
        direct("Material.Full.grain.leather", "a shoe made of full-grain leather showing the natural grain", "Full-grain leather"),
        direct("Material.Sheepskin", "a shoe made of sheepskin", "Sheepskin"),
        direct("Material.Shearling", "a boot lined with shearling, showing thick woolly fleece at the opening", "Shearling"),
        direct("Material.Fleece", "a slipper made of soft fleece", "Fleece"),
        direct("Material.Wool", "a shoe made of wool", "Wool"),
        direct("Material.Felt", "a shoe made of felted wool", "Felt"),
        direct("Material.Satin", "a shoe made of satin with a smooth lustrous sheen", "Satin"),
        direct("Material.Velvet", "a shoe made of velvet with a dense soft pile", "Velvet"),
        direct("Material.Lace", "a shoe made of lace with an openwork patterned fabric", "Lace"),
        direct("Material.Faux.Fur", "a shoe trimmed with imitation fur", "Faux fur"),
        direct("Material.Cork", "a sandal with a cork sole showing a speckled tan texture", "Cork"),
        direct("Material.Jute", "a sandal with a woven jute rope sole", "Jute"),
        direct("Material.Neoprene", "a shoe made of neoprene, a smooth stretchy wetsuit-like rubber fabric", "Neoprene"),
        direct("Material.EVA", "a shoe made of moulded EVA foam", "EVA"),
        described("Material.Nappa", "a shoe made of nappa leather, a soft smooth full-grain leather with a fine even surface", "Nappa"),
        described("Material.Nubuck", "a shoe made of nubuck, a leather sanded to a soft velvety nap, similar to suede but finer", "Nubuck"),
        described("Material.Hair.Calf", "a shoe made of hair-on calfskin, leather with the short animal hair left on, often spotted or striped", "Hair calf"),
        described("Material.Cordura", "a boot made of Cordura, a coarse durable textured nylon fabric used on hiking and work footwear", "Cordura"),
        described("Material.Ripstop", "a shoe made of ripstop nylon, a lightweight fabric with a visible fine square grid woven into it", "Ripstop"),
        described("Material.Microfiber", "a shoe made of microfiber, a fine synthetic fabric with an even matte surface", "Microfiber"),
        described("Material.Terry", "a slipper made of terry cloth with a looped towelling pile", "Terry", notes=TERRY_NOTE),

        direct("ToeStyle.Round Toe", "a shoe with a rounded toe", "Round toe"),
        direct("ToeStyle.Pointed Toe", "a shoe with a sharply pointed toe", "Pointed toe"),
        direct("ToeStyle.Square Toe", "a shoe with a flat square toe", "Square toe"),
        direct("ToeStyle.Open Toe", "a sandal with an open toe leaving the toes exposed", "Open toe"),
        direct("ToeStyle.Closed Toe", "a shoe with a closed toe covering the toes", "Closed toe"),
        direct("ToeStyle.Peep Toe", "a shoe with a small peep-toe opening showing only the tips of the toes", "Peep toe"),
        direct("ToeStyle.Almond", "a shoe with an almond-shaped toe, tapered but softly rounded at the tip", "Almond toe"),
        direct("ToeStyle.Wide Toe Box", "a shoe with a wide roomy toe box", "Wide toe box"),
        direct("ToeStyle.Capped Toe", "a dress shoe with a seam across the toe forming a separate toe cap", "Cap toe"),
        direct("ToeStyle.Wingtip", "a brogue shoe with a W-shaped wingtip seam sweeping back along the sides", "Wingtip"),
        described("ToeStyle.Moc Toe", "a shoe with a prominent U-shaped horseshoe seam across the top of the toe, as if a separate panel of leather were stitched on by hand", "Moc toe", review=REVIEW_CONFIRMED, notes=MOC_ALGONQUIN_NOTE),
        described("ToeStyle.Algonquin", "a shoe with a fine U-shaped seam across the top of the toe forming a flat raised ridge rather than a puckered one", "Algonquin toe", review=REVIEW_CONFIRMED, notes=MOC_ALGONQUIN_NOTE),
        described("ToeStyle.Medallion", "a dress shoe with a decorative pattern of small punched holes on the toe cap, usually floral or circular", "Medallion toe", review=REVIEW_CONFIRMED),
        described("ToeStyle.Bicycle Toe", "a shoe with two parallel seams running down either side of the toe and meeting at the tip like handlebars", "Bicycle toe", review=REVIEW_CONFIRMED),
        described("ToeStyle.Snub Toe", "a shoe with a short blunt rounded toe, stubbier and shorter than a standard round toe", "Snub toe", review=REVIEW_CONFIRMED),
        described("ToeStyle.Bump Toe", "a shoe with a slight raised bump at the very tip of the toe, visible in profile", "Bump toe", review=REVIEW_CONFIRMED),
        described("ToeStyle.Snip Toe", "a western boot with a toe that narrows toward the front but ends in a flat cut tip rather than a point", "Snip toe", review=REVIEW_CONFIRMED),
        described("ToeStyle.Center Seam", "a shoe with a single vertical seam running down the centre of the toe, dividing it symmetrically", "Center seam", review=REVIEW_CONFIRMED),
    ]

    insole = {
        "Leather": "a leather insole", "Padded": "a padded cushioned insole",
        "Removable": "a removable insole", "Moisture.Wicking": "a moisture-wicking insole",
        "Poron": "a Poron foam insole", "EVA": "an EVA foam insole",
        "Textile": "a textile-covered insole", "Orthotic.Friendly": "an orthotic-friendly insole",
        "Memory.Foam": "a memory foam insole", "Polyurethane": "a polyurethane insole",
        "Latex.Lined": "a latex-lined insole", "Synthetic.Leather": "a synthetic leather insole",
        "Gel": "a gel cushioned insole",
    }
    for name, prompt in insole.items():
        entries.append(
            Entry(
                label=f"Insole.{name}",
                prompt=f"a shoe with {prompt}",
                display_en=name.replace(".", " "),
                tier=TIER_DESCRIPTIVE,
                zero_shot_scoreable=False,
                reason=INSOLE_REASON,
                review=REVIEW_SELF,
            )
        )
    return Glossary(entries={e.label: e for e in entries})


if __name__ == "__main__":
    glossary = build()
    glossary.save()
    print(f"wrote {len(glossary.entries)} entries to {Path('locales/glossary.json')}")
