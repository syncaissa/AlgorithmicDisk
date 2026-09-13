-- pandoc Lua filter for the DOCX build of the paper: map the paper's LaTeX environments onto
-- Word paragraph styles defined in reference.docx, and give theorems the numbers the PDF uses.
local defn, prop = 0, 0
local defn_numbers = {"2.1", "2.2"}   -- Definition numbers as printed in the PDF (section.n)
local prop_numbers = {"2.1"}

local function styled_para(text, style)
  return pandoc.Div({pandoc.Para({pandoc.Str(text)})}, pandoc.Attr("", {}, {["custom-style"] = style}))
end

local function renumber(div, word, numbers, counter)
  local p = div.content[1]
  if p and p.t == "Para" and p.content[1] and p.content[1].t == "Strong" then
    local s = p.content[1].content
    if s[1] and s[1].t == "Str" and s[1].text == word then
      counter = counter + 1
      s[3] = pandoc.Str(numbers[counter] or tostring(counter))
    end
  end
  return counter
end

function Div(el)
  local c = el.classes[1]
  if c == "keyinsight" then
    el.attributes["custom-style"] = "Key Insight"
    return {styled_para("Key Insight", "Key Insight Title"), el}
  elseif c == "abstractbox" then
    el.attributes["custom-style"] = "Abstract"
    return {styled_para("Abstract", "Abstract Title"), el}
  elseif c == "keywordsbox" then
    el.attributes["custom-style"] = "Keywords"
    return el
  elseif c == "definition" then
    defn = renumber(el, "Definition", defn_numbers, defn); el.attributes["custom-style"] = "Theorem"; return el
  elseif c == "proposition" then
    prop = renumber(el, "Proposition", prop_numbers, prop); el.attributes["custom-style"] = "Theorem"; return el
  elseif c == "proof" then
    el.attributes["custom-style"] = "Theorem"; return el
  end
end

-- pandoc numbers figure/table captions itself ("Figure 1: ", "Table 1: ") in the same order as LaTeX
