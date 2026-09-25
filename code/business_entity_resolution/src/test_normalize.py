from rapidfuzz.fuzz import token_sort_ratio
import sys
import os

sys.path.insert(0, os.path.abspath("code/business_entity_resolution/src"))
from s1_normalize import normalize_name, normalize_address, get_consonant_skeleton

def test_normalization():
    # 1. "LLC Moncada Léarning Center" and "Moncada Learning Center LLC" -> same core_name
    n1 = normalize_name("LLC Moncada Léarning Center")
    n2 = normalize_name("Moncada Learning Center LLC")
    assert n1[3] == n2[3], f"Expected same core_name, got {n1[3]} and {n2[3]}"
    
    # 2. "Crestline Crestline Clean LP" -> core_name "crestline clean"
    n3 = normalize_name("Crestline Crestline Clean LP")
    assert n3[3] == "crestline clean", f"Expected crestline clean, got {n3[3]}"
    
    # 3. "Pvt. EFS Print Ventures Ltd." -> legal contains private and limited
    n4 = normalize_name("Pvt. EFS Print Ventures Ltd.")
    assert "private" in n4[4] and "limited" in n4[4], f"Expected private and limited in legal, got {n4[4]}"
    
    # 4. "Ectolumdrex dba X+ Madison Inc" -> name_a and name_b both set
    n5 = normalize_name("Ectolumdrex dba X+ Madison Inc")
    assert n5[1] != "" and n5[2] != "", f"Expected name_a and name_b to be set, got {n5[1]} and {n5[2]}"
    
    # 5. "Center 5uperior Co" and "Center Superior Co" -> same name_skel
    n6 = normalize_name("Center 5uperior Co")
    n7 = normalize_name("Center Superior Co")
    assert n6[6] == n7[6], f"Expected same name_skel, got {n6[6]} and {n7[6]}"
    
    # 6. "wilfordhancock.com" -> "wilfordhancock"
    n8 = normalize_name("wilfordhancock.com")
    assert "wilfordhancock" in n8[3] and ".com" not in n8[3], f"Got {n8[3]}"
    
    # 7. address "0200 Washington Street" -> house_no "200"
    a1 = normalize_address("0200 Washington Street")
    assert a1[2] == "200", f"Expected house_no 200, got {a1[2]}"
    
    # "##8 Willow Oak Lane" -> house_no "8", house_masked 1
    a2 = normalize_address("##8 Willow Oak Lane")
    assert a2[2] == "8", f"Expected house_no 8, got {a2[2]}"
    assert a2[3] == 1, f"Expected house_masked 1, got {a2[3]}"
    
    # 8. "1712 Montebello Ave" vs "1712 Montebello Avenue" -> same addr_norm
    a3 = normalize_address("1712 Montebello Ave")
    a4 = normalize_address("1712 Montebello Avenue")
    assert a3[0] == a4[0], f"Expected same addr_norm, got {a3[0]} and {a4[0]}"
    
    # 9. "westfield stone rd" must NOT turn "west" or "stone" into "street"
    a5 = normalize_address("westfield stone rd")
    assert "street" not in a5[0], f"Expected no street, got {a5[0]}"
    
    # 10. one Devanagari name and one Tamil name transliterate to non-empty ASCII
    dev = normalize_name("एसएस फूड प्राइवेट लिमिटेड")
    tam = normalize_name("ராஜ் இன்வெஸ்ட்மெண்ட்ஸ் எல்எல்பி")
    assert dev[3].isascii() and len(dev[3]) > 0, f"Devanagari failed: {dev[3]}"
    assert tam[3].isascii() and len(tam[3]) > 0, f"Tamil failed: {tam[3]}"
    
    # France unit tests
    fra = normalize_address("15 rue de paris 75001")
    assert "rue" in fra[0] and fra[5] == "75001", f"France addr failed: {fra}"
    
    # EDA Pairs unit tests
    # "Sequoia Pony LLC" vs "5equoia Póny LLC" -> high skel or ts ratio
    ts1 = token_sort_ratio(normalize_name("Sequoia Pony LLC")[3], normalize_name("5equoia Póny LLC")[3])
    sk1 = token_sort_ratio(normalize_name("Sequoia Pony LLC")[6], normalize_name("5equoia Póny LLC")[6])
    assert ts1 > 80 or sk1 > 80, f"Sequoia Pony failed: {ts1}, {sk1}"
    
    # "Vijay Life Ventures Private Limited" vs "Private Vijay Lbe Ventures Limited"
    n_v1 = normalize_name("Vijay Life Ventures Private Limited")
    n_v2 = normalize_name("Private Vijay Lbe Ventures Limited")
    sk_v = token_sort_ratio(n_v1[6], n_v2[6])
    assert sk_v > 80, f"Vijay Life failed: {sk_v}"
    
    # "Csk Advisors" vs "CSK ÁDVISORS CORP"
    n_c1 = normalize_name("Csk Advisors")
    n_c2 = normalize_name("CSK ÁDVISORS CORP")
    sk_c = token_sort_ratio(n_c1[6], n_c2[6])
    assert sk_c > 80, f"Csk Advisors failed: {sk_c}"
    
    # "Gavue Resources LLC" vs "Gavue Resources L.L.C."
    n_g1 = normalize_name("Gavue Resources LLC")
    n_g2 = normalize_name("Gavue Resources L.L.C.")
    sk_g = token_sort_ratio(n_g1[6], n_g2[6])
    assert sk_g >= 80, f"Gavue Resources failed: {sk_g}"
    
    # Addresses "1 Franklin Street, Unit 4706, Boston, MA" vs "1 FRANKLIN ST, BOSTON, MA"
    a_f1 = normalize_address("1 Franklin Street, Unit 4706, Boston, MA")
    a_f2 = normalize_address("1 FRANKLIN ST, BOSTON, MA")
    assert a_f1[2] == a_f2[2], f"House no failed: {a_f1[2]} != {a_f2[2]}"
    assert a_f1[6] == a_f2[6], f"State code failed: {a_f1[6]} != {a_f2[6]}"

    print("All unit tests passed!")

if __name__ == "__main__":
    test_normalization()
